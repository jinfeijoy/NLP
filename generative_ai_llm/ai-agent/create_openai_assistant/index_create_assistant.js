import { openai } from './config.js';

// Assistant variables
const asstID = "asst_AlAOsigm1nQkGnahv2eMJBdK";  // this was generated by createAssistant() function

// Upload a file with an "assistants" purpose
const file = await openai.files.create({
  file: await fetch("movies.txt"),
  purpose: "assistants",
});
console.log(file)
// this code will generate the file id can be used in the following code as file id to target the file
// i.e. {object: "file", id: "file-k0Irhj9A6BVEz9kQT6wV7jUz", purpose: "assistants", filename: "movies.txt", bytes: 6176, created_at: 1699718655, status: "processed", status_details: null}


// Create Movie Expert Assistant
async function createAssistant() {
    const myAssistant = await openai.beta.assistants.create({
      instructions: "You are great at recommending movies. When asked a question, use the information in the provided file to form a friendly response. If you cannot find the answer in the file, do your best to infer what the answer should be.",
      name: "Movie Expert",
      tools: [{ type: "retrieval" }],
      model: "gpt-4-1106-preview",
      file_ids: ["file-k0Irhj9A6BVEz9kQT6wV7jUz"] //this file_ids was generated from the above code
    });
  
    console.log(myAssistant);
  }
  createAssistant()

//   this code can generate assistant id can be used for future calling for this assistant 
// {id: "asst_AlAOsigm1nQkGnahv2eMJBdK", object: "assistant", created_at: 1699719024, name: "Movie Expert", description: null, model: "gpt-4-1106-preview", instructions: "You are great at recommending movies. When asked a question, use the information in the provided file to form a friendly response. If you cannot find the answer in the file, do your best to infer what the answer should be.", tools: [{type: "retrieval"}], file_ids: ["file-k0Irhj9A6BVEz9kQT6wV7jUz"], metadata: {}}