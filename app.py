import logging, os, datetime, time
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from openai import OpenAI

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_openai import OpenAI as LLM

load_dotenv()
client = OpenAI()

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/react")
# Choose the LLM to use
llm = LLM(model_name="gpt-3.5-turbo-instruct")

search = GoogleSerperAPIWrapper(type='search')
tavily = TavilySearchResults(max_results=3)
tools = [tavily, Tool(
        name="Serper",
        func=search.run,
        description="useful for when you need to ask with search",
    )]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
# set higher logging level for httpx to avoid all GET and POST requests being logged
# logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    text = "This bot currently contains 2 functionalities.\n\
        1) Use the microphone to send an audio recording message as a prompt for generating an image with DALL-E 3\n\
        2) Enter a text prompt to trigger an agent implementing the ReAct logic with search tools for the incorporation of updated information. Eg: 'Give me three meat items that i can buy on clearance or discount from FairPrice.'"
    # what are the shipping issues with china in the past week?
    # Give me three meat items that i can buy on clearance discount from ntuc
    # where can I buy the cheapest 1kg of frozen brocolli in singapore and what is the price?
    await update.message.reply_text(text)

async def react(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """ReAct"""
    try:
        response = agent_executor.invoke({"input": update.message.text})

        # Print out intermediate steps
        for step in response['intermediate_steps']:
            time.sleep(1.5)
            await update.message.reply_text(step[0].log.split('\n')[0]) # Action
            # Result
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to summarize search results to a maximum of 110 words."},
                    {"role": "user", "content": str(step[1])}
                    ]
                )
            await update.message.reply_text('Observation: {}'.format(summary.choices[0].message.content)) # Action

        # Final output
        await update.message.reply_text('Final Answer: {}'.format(response['output']))
    except:
        await update.message.reply_text("Oops I seem to have exceeded the model's maximum context length. Please try again.")

async def voice(update, context):
    '''Parse audio recording'''
#     # https://docs.python-telegram-bot.org/en/stable/telegram.message.html?highlight=reply_text#telegram.Message.reply_text
    # Get transcript
    audio_file = await context.bot.get_file(update.message.voice.file_id)
    await audio_file.download_to_drive(f"voice_note.ogg")
    await update.message.reply_text('Recording received, image available shortly.')

    with open("voice_note.ogg", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="text"
        )
    await update.message.reply_text('Transcribed: {}'.format(transcript))
    logger.info(transcript)

    # Generate image
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=transcript,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        
        # Return generated image
        await update.message.reply_photo(image_url, caption=transcript)
    except:
        await update.message.reply_text('Issues with generating image')

def main():
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.getenv('TELEGRAM')).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, react))

    # on noncommand i.e message
    application.add_handler(MessageHandler(filters.VOICE, voice))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()