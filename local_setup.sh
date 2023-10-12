# echo "Setting up project..."

# # Create working environment
# echo ">>>Creating working environment"
# #Create a new virtual environment
# python3 -m venv llm_env

# #Activate the virtual environment
# source llm_env/bin/activate
# echo "environment created and activate <<<"

# #Install the dependencies
# echo ">>>Installing dependencies..."
# pip install -r requirements.txt --quiet
# echo "done!<<<"

# # Preprocess the data to create vector stores
# echo ">>>Running data_process.py file<<<"
# python data_process.py

# Download the model from Huggingface
echo ">>>Downloading LLM from huggingface"
wget -O llama-2-7b-chat.ggmlv3.q8_0.bin https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin
echo "done<<<"

# Run the chainlit app, and set the logfile
echo ">>>Running chainlit app.<<<"
chainlit run llm.py -w 
