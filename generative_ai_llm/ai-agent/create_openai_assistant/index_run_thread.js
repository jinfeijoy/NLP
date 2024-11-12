import { openai } from './config.js';

// Assistant variables
const asstID = "asst_AlAOsigm1nQkGnahv2eMJBdK";
const threadID = "thread_XBZNFmOmZvmVWrIHCIfWBU2x";

// List thread messages
async function listMessages() {
  const threadMessages = await openai.beta.threads.messages.list(threadID);

  console.log(threadMessages.data[0].content[0].text.value);
}
listMessages()

// Run the assistant's thread
async function runThread() {
  const run = await openai.beta.threads.runs.create(
    threadID,
    { assistant_id: asstID }
  );
  console.log(run);
}
// runThread()

// Create a message for the thread
async function createMessage() {
  const threadMessages = await openai.beta.threads.messages.create(
    threadID,
    { role: "user", content: "Can you recommend a comedy?" }
  );
  console.log(threadMessages);
}