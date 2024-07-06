import vertexai

from vertexai.generative_models import GenerativeModel, GenerationConfig, ChatSession

# TODO(developer): Update and un-comment below line
# project_id = "PROJECT_ID"

vertexai.init(project='sr2024-farzan', location="us-central1")

model = GenerativeModel(model_name="gemini-1.0-pro-002")

prompt = "Generate a random essay, please make sure it is different from previously generated essays."

for i in range(1,1001):
    fileName = str(i) + ".txt"
    f = open('paragraphs/' + fileName, 'w')
    f.write(model.generate_content(prompt, generation_config = GenerationConfig(temperature=2.0), stream = False).candidates[0].content.parts[0]._raw_part.text)
    f.close()