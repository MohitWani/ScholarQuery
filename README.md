# ScholarQuery

This project is designed to reduce time and get information quickly without wasting time to read research paper. It implement Arxiv AGENT, Tools and Advance Retrieval Augmented Generation to get relevant information about any research papers.

## RAG Process
![Project Image](https://media.beehiiv.com/cdn-cgi/image/fit=scale-down,format=auto,onerror=redirect,quality=80/uploads/asset/file/8368de64-741a-4488-982b-d3e4245334ba/RAG_-_Retrieval.png?t=1709798274)

---

### Key Features.

- Feature 1: Utilize a Langchain Agents and Arxiv TOOLS for getting accurate information about paper.
- Feature 2: Created a Advanced Retrieval Augmented Generation Pipeline for documents Not in Arxiv Library.

---

### Technology used.

- **Language/Frameworks** : Python, Langchain, Streamlit, FastAPI

---

## Installation

Follow these steps to install and run the project locally:

### Prerequisites

Ensure that the following are installed on your system:

#### For Local Setup:

- **Python 3.11.6**: You can download it from [Python's official website](https://www.python.org/downloads/).
- **pip**: Python package installer (comes with Python).
- **git**: For cloning the repository.
- **API keys**: Check .env.example file to add your own api tokens.

#### For Docker Setup:

- Docker (Ensure Docker is installed and running)

### Local Setup

1. **Clone the Repository**

   First, clone the repository from GitHub to your local machine:

   ```bash
   git clone https://github.com/MohitWani/ScholarQuery.git
   ```

2. **Navigate to the Project Directory**

    ```bash
    cd ScholarQuery
    ```
 
3. **Create a Virtual Environment**

    ```bash
    python -m venv env
    env\Scripts\activate
    ```

4. **Install the Required Dependencies**

    install all required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

5. **Setup API TOKENS**

    Create .env file:

    //add your own API tokens use .env.example file for reference.
    ```bash
    touch .env


6. **Run the Application**

    Finally, run the application:

- For a Streamlit app:

    ```bash
    streamlit run app.py
    ```

- For a FastAPI app:

    ```bash
    server.py
    ```

### Docker Setup

1. **Clone the Repository:**

   First, clone the repository from GitHub to your local machine:

   ```bash
   git clone https://github.com/MohitWani/ScholarQuery.git
   ```

2. **Navigate to the Project Directory:**

    ```bash
    cd ScholarQuery
    ```

3. **Build the Docker image:**

    ```bash
    docker build -t scholarquery
    ```

3. **Run the Docker container:**

    ```bash
    docker run -p 8000:8000 -p 8501:8501 scholarquery
    ```

4. **Navigate to Application**

For Server:
    ```bash
    http://localhost:8000/docs
    ```
For Client:
    ```bash
    http://localhost:8501
    ```

---

#### Agent Input and UI
![Project Screenshot](./assets/client1.png)
---
![Project Screenshot](./assets/Agent_output.png)

- Feature 2: Advance Retrieval Augmented Generation for papers not in Arxiv library.

#### RAG Input UI
![Project Screenshot](./assets/client.png)
---
![Project Screenshot](./assets/RAG_output.png)

- Feature 3: FAST API server and REST API.

#### server
![Project Screenshot](./assets/server.png)