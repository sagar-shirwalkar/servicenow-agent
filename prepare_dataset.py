import json

#
# Convert your parsed data into an instruction-tuning 
# format (ChatML or Llama-3 format).
# 

def create_training_data(dataset):
    formatted_data = []
    
    # 1. API Examples (High priority for OpenAPI compliance)
    for api_chunk in dataset['api']:
        formatted_data.append({
            "instruction": "Generate a valid OpenAPI REST request based on the following ServiceNow API documentation.",
            "input": api_chunk,
            "output": "```json\n{\n  \"method\": \"GET\",\n  \"endpoint\": \"/api/now/table/incident\",\n  \"headers\": {\"Accept\": \"application/json\"}\n}\n```"
        })
        
    # 2. Coding Patterns
    for code_chunk in dataset['code']:
        formatted_data.append({
            "instruction": "Write a ServiceNow script to accomplish the task described in the documentation.",
            "input": code_chunk,
            "output": code_chunk # The chunk itself contains the ideal code
        })
        
    # 3. General Q&A (Docs)
    for doc_chunk in dataset['docs'][:500]: # Subsample to balance the dataset
        formatted_data.append({
            "instruction": "Explain the following ServiceNow concept clearly.",
            "input": doc_chunk,
            "output": doc_chunk
        })
        
    with open("train_data.json", "w") as f:
        json.dump(formatted_data, f)
