#!/bin/bash

# Escape newlines and quotes for JSON
SYSTEM_PROMPT=$(cat system_prompt.txt | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')

# Build JSON using files
printf '{
  "messages": [
    {
      "role": "system",
      "content": "%s"
    },
    {
      "role": "user", 
      "content": "Analyze this image and return keypoints in JSON format."
    }
  ],
  "image": "%s"
}' "$SYSTEM_PROMPT" "$(cat image_b64.txt)" > analysis.json

echo "analysis.json built"
