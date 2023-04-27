from app.agent import WebChatAssistant
from app.configs import settings
from app.storage import WebDataExtractor
from app.utils import output_response


if __name__ == "__main__":

    source = WebDataExtractor(
        root_url="https://reldyn.co",
        urls=[
            "https://buy.experian.com.my/index.php/search/Malaysia-Company/1359116/RELDYN-TECH-SDN.-BHD.",
            "https://newday.jobs/job-in-myanmar/junior-senior-ios-guru-developer/70949",
            "https://www.tofler.in/reldyn-tech-private-limited/company/U72900PN2022FTC217062"
        ]
    )
    assistant = WebChatAssistant(source)
    init_conversation = False

    while True:
        try:
            if not init_conversation:
                user_input = input("Hi Am {} Buddy, Ask your question...".format(
                    settings.COMPANY_NAME))
                init_conversation = True
            else:
                user_input = input("Please ask your question...")
            agent_prompt = assistant.prompt_persona.format(
                query=user_input,
                company_name=settings.COMPANY_NAME,
                company_email=settings.COMPANY_EMAIL)
            response = str(assistant.agent.run(agent_prompt)).strip()
            output_response(response)
        except ValueError as e:
            response = str(e)
            response_prefix = "Could not parse LLM output: `"
            if not response.startswith(response_prefix):
                raise e
            response_suffix = "`"
            if response.startswith(response_prefix):
                response = response[len(response_prefix):]
            if response.endswith(response_suffix):
                response = response[:-len(response_suffix)]
            output_response(response)
        except KeyboardInterrupt:
            break
