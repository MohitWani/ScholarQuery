# ScholarQuery (Working)

![Project Image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/8368de64-741a-4488-982b-d3e4245334ba/RAG_-_Retrieval.png?t=1709798274)

## Overview

This project is designed to reduce time and get information quickly without wasting time to read research paper. It implement Arxiv AGENT, Tools and Advance Retrieval Augmented Generation to get relevant information about any research papers.

### Key Features.

- Feature 1: Utilize a Langchain Agents and Arxiv TOOLS for getting accurate information about paper.

![Project Screenshot](D:\my_projects\ScholarQuery\client.png)


- Feature 2: Advance Retrieval Augmented Generation for papers not in Arxiv library.

![Project Screenshot](D:\my_projects\ScholarQuery\client.png)


### Technology used.

- **Language/Frameworks** : Python, Langchain, Streamlit, FastAPI



## Installation

Follow these steps to install and run the project locally:

### Prerequisites

Ensure that the following are installed on your system:

- **Python 3.11.6**: You can download it from [Python's official website](https://www.python.org/downloads/).
- **pip**: Python package installer (comes with Python).
- **git**: For cloning the repository.
- **API keys**: Check .env.example file to add your own api tokens.

### Steps to Install

1. **Clone the Repository**

   First, clone the repository from GitHub to your local machine:

   ```bash
   git clone https://github.com/MohitWani/ScholarQuery.git

2. **Navigate to the Project Directory**

    ```bash
    cd your-repository
 
3. **Create a Virtual Environment**

    ```bash
    python -m venv env
    env\Scripts\activate

4. **Install the Required Dependencies**

    install all required Python packages using pip:
    ```bash
    pip install -r requirements.txt

5. **Setup API TOKENS**

    Create .env file:
    ```bash
    touch .env

    add your own API tokens use .env.example file for reference.

6. **Run the Application**

    Finally, run the application:

    For a Streamlit app:

        ```bash
        streamlit run app.py

    For a FastAPI app:

        ```bash
        server.py