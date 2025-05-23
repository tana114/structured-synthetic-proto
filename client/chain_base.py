import time
from typing import Optional, List, Union, Type, Dict, Any
from typing import cast
from abc import ABCMeta, abstractmethod
import re

import langchain_core.exceptions
import pydantic_core
from pydantic import BaseModel, Field, ValidationError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_core.language_models import BaseChatModel

from langchain.prompts import ChatPromptTemplate
from sympy.liealgebras.type_e import TypeE

""" ***********  invork内容のデバック用 ************ """
import langchain

# # Comment out the following if you want to see the details of the 'invoke' process.
# # invoke時の処理の詳細を確認したい場合は下記をコメントアウト
# langchain.debug = True
""" ********************************************* """

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class ChainBuilder(metaclass=ABCMeta):
    @abstractmethod
    def create_chain(self, **kwargs) -> RunnableSerializable:
        raise NotImplementedError


class SimpleChainBuilder(ChainBuilder):
    def __init__(
            self,
            chat_model: BaseChatModel,
    ):
        self._llm = chat_model

    def create_chain(
            self,
            system_prompt: str,
            human_prompt: str,
    ) -> RunnableSerializable[Dict, str]:  # Dictを受けてstrを返すRunnableを生成
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

        chain = prompt | self._llm | StrOutputParser()
        return chain


class StructuredChainBuilder(ChainBuilder):
    def __init__(
            self,
            chat_model: BaseChatModel,
    ):
        self._llm = chat_model

    def create_chain(
            self,
            system_prompt: str,
            human_prompt: str,
            struct_type: Type[BaseModel],

    ) -> RunnableSerializable[Dict, BaseModel]:  # Dictを受けてBaseModelを返すRunnableを生成

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )

        chain = prompt | self._llm.with_structured_output(struct_type)
        return cast(RunnableSerializable[Dict, BaseModel], chain)


class ChainDirector(object):
    def __init__(
            self,
            chat_model: BaseChatModel,
            system_prompt: str,
            human_prompt: str,
            struct_type: Optional[Type[BaseModel]] = None,
    ):
        if struct_type:
            self._chain = StructuredChainBuilder(chat_model).create_chain(
                system_prompt, human_prompt, struct_type
            )
        else:
            self._chain = SimpleChainBuilder(chat_model).create_chain(
                system_prompt, human_prompt
            )

    def invoke(
            self,
            input: Dict,
            **kwargs,
    ) -> Union[str, BaseModel]:
        return self._chain.invoke(input, **kwargs)


class ConcreteChainBase(metaclass=ABCMeta):
    def __init__(self) -> None:
        self._chain_director: Optional[ChainDirector] = None

    def __call__(
            self,
            input: Dict[str, str],
            **kwargs,
    ):
        return self.invoke(input, **kwargs)

    @abstractmethod
    def _create_chain_director(
            self,
            director_config: Optional[Dict],
    ) -> ChainDirector:
        raise NotImplementedError

    @abstractmethod
    def _invoke_handling(
            self,
            input: Dict[str, str],
            **kwargs,
    ):
        raise NotImplementedError

    def invoke(
            self,
            input: Dict[str, str],
            director_config: Optional[Dict] = None,
            **kwargs,
    ):
        exception_wait_sec = 5
        self._chain_director = self._create_chain_director(director_config)
        # if self._chain_director is None:
        #     logger.error("Chain director is not initialized.")
        #     raise ValueError("Chain director is not initialized.")
        try:

            return self._invoke_handling(input, **kwargs)

        except langchain_core.exceptions.OutputParserException as e:
            logger.error(f"OutputParserException: {e}")
            logger.error(f"Retry invoke {exception_wait_sec} sec after.")
            time.sleep(exception_wait_sec)
            return self.invoke(input, director_config, **kwargs)  # retry
        except (ValidationError, ValueError) as e:
            logger.error(f"ValidationError: {e}")
            logger.error(f"Retry invoke {exception_wait_sec} sec after.")
            time.sleep(exception_wait_sec)
            return self.invoke(input, director_config, **kwargs)  # retry

        except TypeError as e:
            '''
            "groq.BadRequestError"が起きた場合は以下のようなエラーが返される
            APIStatusError.__init__() missing 2 required keyword-only arguments: 'response' and 'body'
            これをgroqモジュールをインポートせずに検出させたい
            '''
            # 正規表現パターン
            pattern = r'APIStatusError'
            match = re.search(pattern, str(e))
            if match:
                logger.error(f"TypeError(APIStatusError): {e}")
                logger.error(f"Retry invoke {exception_wait_sec} sec after.")
                time.sleep(exception_wait_sec)
                return self.invoke(input, director_config, **kwargs)  # retry
            logger.error(f"TypeError: {e}")
            raise
        except Exception as e:
            logger.error(type(e).__mro__)
            logger.error(f"Error: {e}.")
            raise


if __name__ == "__main__":
    """
    python -m client.chain_base
    """

    from logging import DEBUG, INFO, basicConfig

    basicConfig(level=INFO)

    ''' ChainDirector '''

    from model.groq_llm import GroqChatBase


    class ItemData(BaseModel):
        item_name: str = Field(description="商品名")
        price: Optional[int] = Field(None, description="商品の値段")
        color: Optional[str] = Field(None, description="商品の色")


    class ItemList(BaseModel):
        item_list: List[ItemData] = Field(description="複数のItemDataを格納したList")


    llm = GroqChatBase(
        model_name="llama-3.1-8b-instant",
        # requests_per_second=0.32,
        temperature=0,
    )

    client = ChainDirector(
        chat_model=llm,
        system_prompt="与えられた商品の情報を構造化してください:",
        human_prompt="{input}",
        struct_type=ItemList,
    )

    input = "ネクタイは黄色で800円、Tシャツ 赤 142,000円でした"

    res = client.invoke({"input": input})
    print(res)
    # print(res.model_dump())
