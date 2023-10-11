echo "Setting up project..."

# Install the dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "\t\tdone!"

# Preprocess the data to create vector stores
echo "Running data_process.py file"
python data_process.py

# Download the model from Huggingface
echo "Downloading LLM from huggingface"
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin
echo "\t\tdone"

# Install localtunnel
echo "
npm install localtunnel

# # Run the chainlit app, and set the logfile
# chainlit run llm.py -w &>logs.txt &

# # Run the app in localtunnel
# npx localtunnel --port 8000
