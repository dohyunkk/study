
#! State, Node, Edge, Reducer

from typing import Annotated, TypedDict    #* dictionaly type
from langgraph.graph import add_messages

class Mydate(TypedDict):
    messages : Annotated[list, add_messages]   # Reducer
    
from langchain_core.messages import AIMessage, HumanMessage

msg1 = [HumanMessage(content='안녕', id='1')]
msg2 = [HumanMessage(content='방가', id='2')]

results = add_messages(msg1, msg2)
print(results)

#* [HumanMessage(content='안녕', additional_kwargs={}, response_metadata={}, id='1'), HumanMessage(content='방가', additional_kwargs={}, response_metadata={}, id='2')]

##################################################
msg3 = [HumanMessage(content='안녕', id='3')]
msg4 = [HumanMessage(content='방가', id='3')]

results2 = add_messages(msg3, msg4)
print(results2)

[HumanMessage(content='방가', additional_kwargs={}, response_metadata={}, id='3')]

