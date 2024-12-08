# Semantic-Embedding-for-Mood-Based-Yoga-Session-Recommendation-and-Report-Generation
This repo shows the codes for Semantic Embedding for Mood-Based Yoga Session Recommendation and Report Generation

---


# Ollama Installation

curl -fsSL https://ollama.com/install.sh | sh

Then use required LLMs and Embeddings from Ollama

# Setup

1. Python3 venv -m yoga
2. source yoga/bin/activate
3. pip install -r requirements


# Run the Server Code on FastAP
Python3 mood_9.py

# Run the Client Code

From same project directory run

python3 curl_caller_mood.py

# Result 

1. A directory called **reports** is generated with PDF reports based on mood which was given in the prompts as mentioned in **curl_caller_mood.py**
2. A CSV file for metrics analysis is generated **yoga_asana_metrics.csv**

---

## Detailed Step-by-Step Workflow

---

### **1. Application Setup**

#### **1.1 Initialize FastAPI Application**
- Instantiate a `FastAPI` object to manage routing and endpoint handling.
- Define endpoints for processing user prompts and serving PDF reports.

#### **1.2 Configure Logging**
- Set logging to the `DEBUG` level to capture detailed logs.
- Use `StreamHandler` to ensure logs are visible in the console during runtime.
- Log all key operations and errors for debugging and tracking.

#### **1.3 Define Configuration Variables**
- **Embedding Model**: Choose a pre-trained embedding model (`mxbai-embed-large`) to compute vector representations of text.
- **Similarity Threshold**: Set a similarity threshold (`0.62`) to determine if an asana matches the user prompt.
- **API Endpoints**:
  - Ollama API for embedding generation.
  - Ollama API for generating comments via a large language model (LLM).
- **Report Directory**: Create a folder named `reports` to store generated PDF reports.
- **CSV Metrics File**: Define a CSV file (`yoga_asana_metrics.csv`) to log request metrics.

---

### **2. Data Initialization**

#### **2.1 Predefined Yoga Asanas**
- Define a list of yoga asanas, each containing:
  - **Name**: Name of the asana.
  - **Utterances**: Sample phrases representing moods the asana addresses.
  - **Details**:
    - Step-by-step instructions.
    - Recommended frequency and timing.
    - Dietary and lifestyle recommendations.
    - Benefits.

#### **2.2 Embedding Cache**
- Create a dictionary (`cached_embeddings`) to store embeddings of predefined asana utterances.
- Use the cache to avoid redundant API calls for embeddings.

---

### **3. Startup Tasks**

#### **3.1 Load Embedding Model**
- On application startup:
  - Test the embedding API by generating an embedding for a sample text.
  - Log success or failure of the embedding model.

#### **3.2 Cache Asana Utterance Embeddings**
- Precompute embeddings for all utterances in the predefined asanas using the embedding API.
- Store these embeddings in the `cached_embeddings` dictionary for efficient retrieval during requests.

---

### **4. Request Handling**

#### **4.1 Endpoint: `/process_prompt`**

**Step 1: Receive User Input**
- Accept a JSON payload containing a prompt (e.g., "I feel overwhelmed and stressed with my work").
- Parse and validate the input using `Pydantic`.

**Step 2: Generate Prompt Embedding**
- Send the user prompt to the embedding API to retrieve its semantic vector representation.
- If the embedding generation fails, return a `500 Internal Server Error` response.

**Step 3: Find the Best Asana**
- Iterate through the predefined asanas and compute the cosine similarity between the prompt embedding and the cached embeddings of asana utterances.
- Select the asana with the highest similarity score above the threshold (`0.62`).
- Measure and log the time taken for this matching process.
- **No Match Handling**:
  - If no asana meets the similarity threshold:
    - Log the event with `no_route`.
    - Save metrics to the CSV file.
    - Return a "no match" response to the user.

**Step 4: Generate Final Comment**
- Use the LLM API to generate a personalized comment about the selected asana.
- Pass a few-shot example (predefined input-output pairs) and the selected asana's details to the API for context.
- If the LLM API fails, use a fallback message and log the error.

**Step 5: Generate PDF Report**
- Use the `ReportLab` library to create a PDF report containing:
  - User prompt and similarity score.
  - Detailed instructions for the recommended asana.
  - Dietary and lifestyle recommendations.
  - LLM-generated final comment.
- Save the PDF in the `reports` directory.
- If PDF generation fails, return a `500 Internal Server Error` response.

**Step 6: Log Metrics**
- Record the following details in the CSV file:
  - Date and time of the request.
  - Selected asana and similarity score.
  - LLM metrics (response time, tokens processed, etc.).
  - PDF generation time and embedding match duration.

**Step 7: Return Response**
- Respond to the user with:
  - Selected asana details.
  - LLM-generated final comment.
  - Download link for the PDF report.
  - Total processing time.

---

### **5. Report Download**

#### **5.1 Endpoint: `/download_report/{report_filename}`**
- Accept a filename as input.
- Locate the corresponding PDF file in the `reports` directory.
- Return the file as a downloadable response.
- **Error Handling**:
  - Return a `404 Not Found` response if the file doesn’t exist.

---

### **6. Utility Functions**

#### **6.1 Embedding Generation**
- Send a POST request to the embedding API with the prompt or utterance.
- Parse and return the embedding vector from the API response.

#### **6.2 Cosine Similarity Calculation**
- Compute the similarity between two embeddings using:
  \[
  \text{Cosine Similarity} = \frac{\text{dot product of vectors}}{\text{product of vector magnitudes}}
  \]

#### **6.3 Find Best Asana**
- Compare the prompt embedding against cached embeddings for all asana utterances.
- Return the asana with the highest similarity score above the threshold.

#### **6.4 Generate Final Comment**
- Send the selected asana's details and user prompt to the LLM API.
- Use few-shot learning to guide the LLM in generating concise, context-aware comments.

#### **6.5 Generate PDF Report**
- Use the `ReportLab` library to create a visually structured PDF with:
  - Header: Mood-Based Yoga Recommendation Report.
  - User prompt and similarity score.
  - Detailed instructions and recommendations for the asana.
  - Final comment.
- Save the report in the `reports` directory.

---

### **7. Metrics Logging**
- Record key metrics for each request:
  - **Request Details**: User prompt, selected asana.
  - **Performance Metrics**:
    - Embedding match duration (logged as `embed_match_duration` in nanoseconds).
    - LLM response time and tokens processed.
    - PDF generation time.
  - Save these metrics to the `yoga_asana_metrics.csv` file.

---

### **8. Error Handling**
- Handle exceptions at every step:
  - Log errors with detailed messages.
  - Return user-friendly HTTP error responses:
    - `500` for internal server issues.
    - `404` for missing files.

---

### **9. Application Deployment**
- Start the FastAPI server using `uvicorn` on port `5000`.
- Test endpoints using tools like `curl` or Postman to ensure end-to-end functionality.

---

### Sample PDF Report on Mood Based Yoga Asana 

![image](https://github.com/user-attachments/assets/184291fe-3d64-4723-9f34-157b2ef75929)



