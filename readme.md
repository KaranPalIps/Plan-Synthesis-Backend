Plan Synthesis Backend README
Welcome to the Plan Synthesis backend project. This project implements a Retrieval-Augmented Generation (RAG) application using the LLama Index. The instructions below will guide you through the process of setting up, configuring, and running the project.

Setup Instructions
1. Create a Virtual Environment
To ensure a clean and isolated environment, create a virtual environment in your project directory:


python -m venv venv
2. Activate the Virtual Environment
Activate the virtual environment you just created:

For Windows:
.\venv\Scripts\activate

For macOS/Linux:
source venv/bin/activate

3. Install Dependencies
With the virtual environment active, install all the required dependencies from requirements.txt:


pip install -r requirements.txt

4. Obtain a Groq API Key
To access Groq services, you need an API key:

Go to Groq Cloud Console.
Log in or create a new account.
Generate an API key and copy it.
5. Set the API Key in the Environment File
In the root directory of the project, create a file named .env if it doesn't already exist. Add your Groq API key to this file with the key name GROQ_API_KEY:


echo "GROQ_API_KEY=your_api_key_here" > .env

6. Run the Code
With the virtual environment activated and all dependencies installed, you can now start the server. Use uvicorn to run the server and reload it on code changes:

uvicorn server:app --reload


This command will start the server, and you can access the application at http://127.0.0.1:8000/.

Contributing
If you'd like to contribute to this project, please submit a pull request with a clear explanation of your changes. Ensure that your code adheres to the existing style and conventions.

License
This project is licensed under the [License Name] License - see the LICENSE file for details.

Thank you for contributing to the Plan Synthesis backend project. If you have any questions or issues, feel free to reach out or open a GitHub issue.