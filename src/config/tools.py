from model.topics_model import Topics
from langchain_core.tools import tool


@tool
def create_list_of_topics(list_of_topics: list[str]) -> Topics:
    """
    you should give a list of strings and this funciton will return a list of topics

    :param list_of_topics: a list of topics
    :return: Topic list
    """
    return Topics(topics=list_of_topics)

@tool
def create_summarization():
    """

    :return:
    """
    pass