from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.chains.conversation.memory import ConversationBufferMemory
from llama_index import QueryMode
from langchain.llms import OpenAI

from app.configs import settings
from app.storage import BaseDataSource


class WebChatAssistant(object):

    def __init__(self,  data_source: BaseDataSource):
        """
        Initializes a new instance of the WebChatAssistant class. to provide answers to questions.

        Args:
            data_source: An instance of a class that inherits from BaseDataSource.
               The data source provides the index for the chat assistant to search for answers.

        Raises:
            TypeError: If data_source is not an instance of BaseDataSource.

        Returns:
            A new instance of the WebChatAssistant class.
        """
        if not isinstance(data_source, BaseDataSource):
            raise TypeError("data_source must be an instance of BaseDataSource")
        self._index = data_source.index()
        self.llm = OpenAI(
            model_name=settings.OPENAI_MODEL.name,
            temperature=settings.OPENAI_MODEL.temperature
        )
        self.tools = [
            Tool(
                name="Website Index",
                func=lambda q: str(self.index.query(q, mode=QueryMode.EMBEDDING)),
                description="""
                useful for when you want to answer questions from Website Index. Always, 
                you must try the index first, only answer based this Website Index
                """,
            )
        ]

    @property
    def memory(self) -> ConversationBufferMemory:
        return ConversationBufferMemory(memory_key="chat_history")

    @property
    def agent(self) -> AgentExecutor:
        """
        This function returns an instance of an AgentExecutor
        class initialized with specific parameters. The AgentExecutor class is a part of a
        conversational AI system that can interact with users and provide responses.
        The function calls the initialize_agent() method to create
        the AgentExecutor object with the following parameters:

        self.tools:
            a set of tools required for the AgentExecutor object to operate.
        self.llm:
            a language model used by the agent to generate responses.
        agent:
            This agent uses the ReAct framework to determine which tool to use based
            solely on the tool’s description. Any number of tools can be provided. This agent requires that a
            description is provided for each tool
        memory:
            a memory module that allows the agent to remember past conversations.
            verbose: a flag that determines whether the agent's internal messages should be printed to the console.
        max_iterations:
            the maximum number of iterations the agent can perform during a single conversation.

        Returns: AgentExecutor object can be used to interact with users and provide responses to their queries.
        """

        return initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            max_iterations=10
        )

    @property
    def prompt_persona(self) -> PromptTemplate:
        return PromptTemplate(
            template="""
            You are a personal assistant for {company_name} company your job is to answer questions. 
            Use only context Website Index to provide answers.
            Do not provide any answers that deviate from your tools documents.
            If you don't know the answer, just say "Hmm, Im not sure please contact customer support at {company_email} 
            for further assistance." Don't try to make up an answer.:
            --------
            Question: {query}
            """,
            input_variables=["query", "company_name", "company_email"],
        )

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
