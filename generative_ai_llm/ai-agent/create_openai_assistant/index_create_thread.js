import { openai } from './config.js';

// Assistant variables
const asstID = "asst_AlAOsigm1nQkGnahv2eMJBdK";
const threadID = "thread_XBZNFmOmZvmVWrIHCIfWBU2x";

// Create a message for the thread
async function createMessage() {
  const threadMessages = await openai.beta.threads.messages.create(
    threadID,
    { role: "user", content: "Can you recommend a comedy?" }
  );
  console.log(threadMessages)
}
createMessage()

// Create a thread
// const thread = await openai.beta.threads.create();
// console.log(thread)