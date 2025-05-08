import json
import os

def extract_title_components(title):
    try:
        # Handle cases where title might not have expected format
        form_part = "Unknown Form"
        topic_part = "Unknown Topic"
        sub_topic = "Unknown Sub-topic"
        
        # Split by colon first
        if ":" in title:
            form_part, rest = title.split(":", 1)
            form_part = form_part.strip()
            rest = rest.strip()
            
            # Then try to split by hyphen
            if " - " in rest:
                topic_part, sub_topic = rest.split(" - ", 1)
            else:
                # If no hyphen, consider everything after colon as topic
                topic_part = rest
                sub_topic = ""
                
            topic_part = topic_part.strip()
            sub_topic = sub_topic.strip()
        
        return {
            "form": form_part,
            "topic": topic_part,
            "sub_topic": sub_topic
        }
    except Exception as e:
        print(f"Warning: Error processing title '{title}': {str(e)}")
        return {
            "form": "Unknown Form",
            "topic": "Unknown Topic" ,
            "sub_topic": "Unknown Sub-topic"
        }

def process_json_file():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "notes_examples_questions.json")
    output_file = os.path.join(script_dir, "processed_content.json")
    
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # List to store all processed entries
        processed_entries = []
        
        # Process each object in the JSON file
        for item in data:
            if "notes" not in item or "title" not in item["notes"]:
                print("Warning: Skipping item without notes or title")
                continue
                
            # Extract components from the title
            title_components = extract_title_components(item["notes"]["title"])
            
            # Process notes section
            notes_entry = {
                **title_components,
                "content_type": "notes",
                "content": item["notes"]
            }
            processed_entries.append(notes_entry)
            
            # Process worked examples section if it exists
            if "worked_examples" in item:
                worked_examples_entry = {
                    **title_components,
                    "content_type": "worked_examples",
                    "content": item["worked_examples"]
                }
                processed_entries.append(worked_examples_entry)
            
            # Process questions section if it exists
            if "questions" in item:
                questions_entry = {
                    **title_components,
                    "content_type": "questions",
                    "content": item["questions"]
                }
                processed_entries.append(questions_entry)
        
        # Write the processed entries to a new JSON file
        with open(output_file, 'w') as f:
            json.dump(processed_entries, f, indent=2)
            
        print(f"Successfully processed {len(processed_entries)} entries.")
        print(f"Output written to: {output_file}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    process_json_file() 