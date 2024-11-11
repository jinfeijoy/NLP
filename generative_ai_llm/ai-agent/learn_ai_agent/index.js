/* Agent Setup*/
import OpenAI from "openai"

export const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    dangerouslyAllowBrowser: true
})

/**
 * Goal - build an agent that can get the current weather at my current location
 * and give me some localized ideas of activities I can do.
 */

const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
        {
            role: "user",
            content: "Give me a list of activity ideas based on my current location and weather"
        }
    ]
})

console.log(response.choices[0].message.content)
/**
As an AI, I'm unable to access your current location or weather details. However, I can provide you with activities for various scenarios: 1. If it's sunny and warm: - Visit the beach - Go on a picnic - Try water sports - Hiking or trail running - Visit a farmers market or outdoor festival 2. If it's sunny but cold: - Go skiing or snowboarding - Ice skating - Take scenic photos - Visit a museum or indoor gallery - Try out a local coffee shop or bakery 3. If it's raining: - Visit a local museum or art gallery - Go to a movie theater - Have a cozy day in with your favorite books and movies - Try a new indoor hobby, like painting or baking 4. If it's cloudy or overcast: - Visit a botanical garden or aquarium - Go on a city exploration walk - Try out a cooking class - Visit a historic site or museum. Remember to check local guidelines and restrictions due to Covid-19 before finalizing any plans.
 */

/* Introduction to react prompting*/
// Call a function to get the current location and the current weather
// to create tools.js file to include some functions can be called
const weather = await getCurrentWeather()
const location = await getLocation()

const response = await openai.chat.completions.create({
    model: "gpt-4",
    messages: [
        {
            role: "user",
            content: `Give me a list of activity ideas based on my current location of ${location} and weather of ${weather}`
        }
    ]
})

console.log(response.choices[0].message.content)
/**
 * 1. Visit Salt Lake City Public Library: Take advantage of the sunny weather to walk around this architectural masterpiece and the vibrant community gathering place. 2. Explore Antelope Island State Park: Watch wild bison and hike the trails. It's just the perfect day with 72°F to explore the natural wonders of the area. 3. Head to Sugar House Park: You can run, walk, or jog around the area. Don't forget to have a picnic under the sun. 4. Visit Temple Square: Rich with history, architecture, and stunning landscaped gardens, Temple Square is an iconic location in the city. 5. Stroll around Red Butte Garden: Enjoy the mild weather in this botanical garden, arboretum, and amphitheater. 6. Enjoy the Outdoors at Liberty Park: Perfect weather to rent a paddle boat, visit the aviary or even just have a barbeque. 7. Visit the Natural History Museum of Utah: If you feel like taking a break from the sunshine, step inside this museum to learn about Utah's natural history. 8. Hiking Big Cottonwood Canyon: Keep your water bottle and sunscreen handy and enjoy the hike under the clear skies. 9. Bike the Jordan River Parkway Trail: Enjoy a leisurely cycle along this scenic path. 10. Salt Lake City Farmer's Market: Take advantage of the beautiful weather to purchase some local produce or handmade goods and enjoy local live performances. 11. Take a Ride on the Salt Lake Trolley: Sightsee Salt Lake City's attractions under the comfortable weather. 12. Explore Bonneville Shoreline Trail: Perfect for biking, hiking or running with a beautiful view over the surroundings. 13. Visit Hogle Zoo: A sunny and warm day is perfect to explore more than 800 animals from all over the world. 14. Golfing: With the sunny weather, why not go out and play some Golf in any of the immaculate courses around the city. 15. Paddleboarding or Kayaking in Great Salt Lake: Make the most of this perfect weather by enjoying water sports in the Great Salt Lake. Make sure to follow any local guidelines or restrictions due to the ongoing pandemic situation. Enjoy your day in Salt Lake City!
/index.html
 */

/** Write ReAct prompt - planning */
/**
 * Goal - build an agent that can answer any questions that might require knowledge about my current location and the current weather at my location.
 */

/**
 PLAN:
 1. Design a well-written ReAct prompt
 2. Build a loop for my agent to run in.
 3. Parse any actions that the LLM determines are necessary
 4. End condition - final Answer is given
  */

 const systemPrompt = `
 You cycle through Thought, Action, PAUSE, Observation. At the end of the loop you output a final Answer. Your final answer should be highly specific to the observations you have from running
 the actions.
 1. Thought: Describe your thoughts about the question you have been asked.
 2. Action: run one of the actions available to you - then return PAUSE.
 3. PAUSE
 4. Observation: will be the result of running those actions.
 
 Available actions:
 - getCurrentWeather: 
     E.g. getCurrentWeather: Salt Lake City
     Returns the current weather of the location specified.
 - getLocation:
     E.g. getLocation: null
     Returns user's location details. No arguments needed.
 
 Example session:
 Question: Please give me some ideas for activities to do this afternoon.
 Thought: I should look up the user's location so I can give location-specific activity ideas.
 Action: getLocation: null
 PAUSE
 
 You will be called again with something like this:
 Observation: "New York City, NY"
 
 Then you loop again:
 Thought: To get even more specific activity ideas, I should get the current weather at the user's location.
 Action: getCurrentWeather: New York City
 PAUSE
 
 You'll then be called again with something like this:
 Observation: { location: "New York City, NY", forecast: ["sunny"] }
 
 You then output:
 Answer: <Suggested activities based on sunny weather that are highly specific to New York City and surrounding areas.>
 `
 