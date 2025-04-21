from typing import List, Union

from qdrant_client.http import models


def matchany_filter(
    data_key: str, match_criteria: Union[List[str], List[int]]
):
    return models.Filter(
        should=[
            models.FieldCondition(
                key=data_key, match=models.MatchAny(any=match_criteria)
            )
        ]
    )
