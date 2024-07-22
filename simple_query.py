import os
from typing import cast

import pandas
import vertexai
from google.cloud import bigquery
from vertexai.generative_models import (
    Content,
    FunctionDeclaration,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    Part,
    Tool,
    ToolConfig,
)

GEMINI_PROJECT_ID = os.getenv("GEMINI_PROJECT_ID")
GEMINI_LOCATION = "asia-northeast1"
GEMINI_MODEL_NAME = "gemini-1.5-flash-001"

BIGQUERY_JOB_PROJECT_ID = os.getenv("BIGQUERY_JOB_PROJECT_ID")
BIQQUERY_JOB_LOCATION = "us"

vertexai.init(project=GEMINI_PROJECT_ID, location=GEMINI_LOCATION)


def run_query(query: str) -> pandas.DataFrame:
    """
    Get information from data in BigQuery using SQL queries

    Args:
        query (str): SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names.
    Returns:
        Dict[str, Any]: Query result
    """
    print(query)
    client = bigquery.Client(
        project=BIGQUERY_JOB_PROJECT_ID, location=BIQQUERY_JOB_LOCATION
    )
    resp = client.query_and_wait(query)
    return resp.to_dataframe()


def generative_model():
    return GenerativeModel(
        GEMINI_MODEL_NAME,
        generation_config=GenerationConfig(
            temperature=0,
        ),
        tool_config=ToolConfig(
            ToolConfig.FunctionCallingConfig(
                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
            ),
        ),
        tools=[
            Tool(
                [
                    FunctionDeclaration(
                        name="run_query",
                        description="Get information from data in BigQuery using SQL queries",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "SQL query on a single line that will help give quantitative answers to the user's question when run on a BigQuery dataset and table. In the SQL query, always use the fully qualified dataset and table names.",
                                }
                            },
                            "required": [
                                "query",
                            ],
                        },
                    ),
                ]
            )
        ],
    )


def main():
    model = generative_model()

    user_content = Content(
        role="user",
        parts=[
            Part.from_text(
                "今日の日付を取得するクエリを実行し、今日が何日か教えて下さい。"
            )
        ],
    )
    response = cast(
        GenerationResponse,
        model.generate_content(
            [user_content],
            generation_config=GenerationConfig(
                temperature=0,
            ),
        ),
    )

    if len(response.candidates[0].function_calls) > 0:
        function_call = response.candidates[0].function_calls[0]
        fc_responses = []
        if function_call.name == "run_query":
            query = str(function_call.args["query"])
            results = run_query(query)
            for col in results.columns:
                if results[col].dtype == "dbdate":
                    results[col] = results[col].map(
                        lambda x: x.strftime("%Y-%m-%dT%H:%M:%S%Z")
                    )
            fc_responses = results.to_dict(orient="records")

        final_response = cast(
            GenerationResponse,
            model.generate_content(
                [
                    user_content,
                    response.candidates[0].content,
                    Content(
                        parts=[
                            Part.from_function_response(
                                name="run_query",
                                response={
                                    "contents": fc_responses,
                                },
                            ),
                        ],
                    ),
                ],
            ),
        )

        print(final_response.candidates[0].content.parts[0].text)


if __name__ == "__main__":
    main()
