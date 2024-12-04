## Learn AI Agents
_ai agent: like a project manager to take actions and made decisions based on contect to achieve a specific goal or set of goals. the agent operates autonomously, meaning it is not directly controlled by a  human operator_
* prompt engineering / chain-of-thought prompting: is the art of getting AI to do what you want
    * be specific / Use technical terms / provide context / give examples / iterate 
    * specify the length of responses / specify the format of responses / 
* agents
    * LLM /text to speech / image generation / CV
    * setup
* agent tools/functions
    * chain: call function to get some data -> pass data to LLM to reason about -> get response from LLM
* building a ReAct Agent (reasoning + acting)
    * reasoning: think about what steps or additional information you might require to accomplish your task
    * acting: perform an action to help you get closer to your ultimate goal
    * observing: observe the results of the action you performed, and start back at reasoning when necessary, otherwise, give the completed response to the user
    * ![image](https://github.com/user-attachments/assets/718ceec7-106b-484f-82f4-c56aa2db6fdc)
    * ![image](https://github.com/user-attachments/assets/21f652bb-cf46-4297-9eb7-c1bc86ba57bb)
    * [A simple Python implementation of the ReAct pattern for LLMs](https://til.simonwillison.net/llms/python-react-pattern)
    * user need to specifit prompt for ReAct
* Building an OpenAI Functions Agent
    * [openai function calling](https://platform.openai.com/docs/guides/function-calling)
    * [api reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools)
    * no need to define own prompt for loop
* Automatic Function Calling 
    * [automatic function call](https://github.com/openai/openai-node/tree/master#automated-function-calls)

## Learn OpenAI's [Assistant API](https://platform.openai.com/docs/assistants/overview)
* [login page](https://platform.openai.com/playground/assistants?mode=assistant)
* retrieve external knowledge / manage threads / execute functions / interpret code / make your ai more responsive and informed
* overview
    * code interpreter: run code and test code
    * knowledge retrieval: pull eternal knowledge from outside its model
    * function calling
    * ![image](https://github.com/user-attachments/assets/6bda1019-10b8-499b-a4c2-2425a2327cd2)
* create an assistant
* give assistant access to files
* create conversations with threads & messages 

## Build AI Apps with LangChain.js
_[LangChain](https://js.langchain.com/docs/introduction/) is a framework that helps developers build context-aware reasoning applications._
* app flow diagrams
    * ![image](https://github.com/user-attachments/assets/05388cf7-738d-4197-baed-118bad0c7877)
       * information source: knowledge for the app
       * splitter: langchain tool to split document to chuncks
       * embeddings: embedding chuncks
       * [supabase](https://supabase.com/) vector store: store chuncks
            * create project with name and db pwd
            * create table and search function in database (with new query) 
       * split the text
       * upload to supabase
    * ![image](https://github.com/user-attachments/assets/4760736a-4e8e-43a6-9782-8212dd4dc965)
       * user input
       * conversation memory: to hold entire conversation
       * openai to be used to create standalone question with no unnecessary words
       * 
      
* embedding
* vector stores
* templates
* prompts from templates
* setting up chains
    * adding first chain
    * retrieval
    * add StringOutputParser
    * fetching the answer
    * serrialize the docs
* the .pipe() method
* retrieving from vector stores
* the RunnableSequence class

## AI Agents in LangGraph 
