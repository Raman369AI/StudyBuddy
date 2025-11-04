from llama_index.llms.google_genai import GoogleGenAI
import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex,SimpleDirectoryReader, Settings,StorageContext

# Load environment variables

from logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from llama_index.core import SummaryIndex
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.schema import TransformComponent
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
)
from llama_index.core.node_parser import SemanticSplitterNodeParser
import qdrant_client
from llama_index.core.node_parser import (SemanticSplitterNodeParser, SentenceSplitter)
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel, ConfigDict,Field
from fastapi import FastAPI, HTTPException,File,UploadFile,Form
import requests
import fitz
from fastapi.responses import JSONResponse
import json
import tempfile
import asyncio
from fastapi.background import BackgroundTasks
import itertools
from fastapi.middleware.cors import CORSMiddleware
import traceback
from typing import List, Optional
from helper_functions import extract_text_from_pdf,extract_text_from_word,cleanup_temp_dir
import re
import json
import pickle
import hashlib
from datetime import datetime

def fix_json_string(invalid_json_str):
    # Remove any leading/trailing whitespace
    json_str = invalid_json_str.strip()
    
    # Fix the options format - this is the main issue
    # Look for "options": followed by key-value pairs without proper JSON formatting
    pattern = r'"options":\s*"([A-Z])":\s*"([^"]+)"(?:,\s*"([A-Z])":\s*"([^"]+)")*'
    
    # Function to replace each match with properly formatted JSON
    def replace_options(match):
        options_text = match.group(0)
        # Extract all option pairs using regex
        option_pairs = re.findall(r'"([A-Z])":\s*"([^"]+)"', options_text)
        
        # Format as proper JSON object
        formatted_options = '"options": {'
        formatted_options += ', '.join([f'"{key}": "{value}"' for key, value in option_pairs])
        formatted_options += '}'
        
        return formatted_options
    
    # Apply the fix
    fixed_json = re.sub(pattern, replace_options, json_str)
    
    # Validate the fixed JSON
    try:
        parsed = json.loads(fixed_json)
        return fixed_json, True
    except json.JSONDecodeError as e:
        return f"Still invalid JSON: {str(e)}", False


    
    # Convert to JSON string
    valid_json = json.dumps(questions, indent=2)
    return valid_json


app = FastAPI(debug=os.getenv("DEBUG_MODE", "false").lower() == "true")

# origins = [
#     "http://localhost:9002",  #  The origin of your React app (port 3000 is common for Create React App)
#     "http://127.0.0.1:9002", #  Include this as well
# ]





origins = os.getenv("ALLOWED_ORIGINS", "https://localhost:9002,https://127.0.0.1:9002").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],  #  Allow all HTTP methods (POST, GET, etc.)
    allow_headers=["*"],  #  Allow all headers
)

class MCQ(BaseModel):
    question: str
    options: dict[str, str]
    correct_answer: str

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

class MCQ_list(BaseModel):
    MCQ_list: list[MCQ]

def rev_valid_json_string(response):
    # Remove newlines and extra spaces
    valid_json_string = re.sub(r"\n", '', response)
    valid_json_string = re.sub(r"```json", '', valid_json_string)
    valid_json_string = re.sub(r"```", '', valid_json_string)
    valid_json_string = re.sub(r"   ", '', valid_json_string)
    return valid_json_string

class UserRelatedData(BaseModel):
    user_query: Optional[str] = Field(default='', description="Any specific topics the user wants to view questions for or any additional info about the questions he wants to see ")
    num_of_questions: int = Field( ge=0, description="The number of questions the user wants to see")

class ProcessingSession(BaseModel):
    session_id: str
    total_chunks: int
    processed_chunks: int
    all_mcqs: List[dict]
    total_questions_generated: int
    target_questions: int
    created_at: datetime
    last_updated: datetime

def generate_session_id(files_content: bytes, user_query: str, num_questions: int) -> str:
    """Generate a unique session ID based on file content and parameters"""
    content_hash = hashlib.md5(files_content + user_query.encode() + str(num_questions).encode()).hexdigest()
    return f"session_{content_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def save_session(session: ProcessingSession):
    """Save session state to file"""
    session_dir = "processing_sessions"
    os.makedirs(session_dir, exist_ok=True)
    with open(f"{session_dir}/{session.session_id}.pkl", "wb") as f:
        pickle.dump(session, f)

def load_session(session_id: str) -> Optional[ProcessingSession]:
    """Load session state from file"""
    session_file = f"processing_sessions/{session_id}.pkl"
    if os.path.exists(session_file):
        with open(session_file, "rb") as f:
            return pickle.load(f)
    return None

def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    session_dir = "processing_sessions"
    if not os.path.exists(session_dir):
        return
    
    cutoff_time = datetime.now().timestamp() - (int(os.getenv("SESSION_CLEANUP_HOURS", "24")) * 60 * 60)
    for filename in os.listdir(session_dir):
        filepath = os.path.join(session_dir, filename)
        if os.path.getctime(filepath) < cutoff_time:
            os.remove(filepath)



@app.post("/resume-processing/")
async def resume_processing(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...)
):
    try:
        # Load existing session
        session = load_session(session_id)
        if not session:
            return JSONResponse(content={"error": "Session not found or expired"}, status_code=404)
        
        # Continue processing from where we left off
        return await continue_processing(session, background_tasks)
        
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

@app.post("/process-user-data/")
async def process_user_data(
    background_tasks: BackgroundTasks,
    user_query: str = Form(...),
    num_of_questions: int = Form(...),
    files: List[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
 
    # Clean up old sessions
    cleanup_old_sessions()
    
    # If resuming an existing session
    if session_id:
        existing_session = load_session(session_id)
        if existing_session:
            return await continue_processing(existing_session, background_tasks)
    
    temp_dir = tempfile.mkdtemp()
    temp_file_paths = []

    try:
        # 2. Save uploaded files to the temporary directory
        if files:
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                temp_file_paths.append(file_path)

        # 3. Use SimpleDirectoryReader to load all files in the temp directory
        
        # Create Patient object
        user_related_data = UserRelatedData(
            user_query=user_query,
            num_of_questions=num_of_questions
        )
        
        print("User data:", user_related_data.dict())

        client = qdrant_client.QdrantClient(location=":memory:")
        vector_store = QdrantVectorStore(client=client, collection_name="rag_llm")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        llm = GoogleGenAI(
    model=os.getenv("LLM_MODEL", "gemini-2.0-flash"),
    vertexai_config={"project": os.getenv("GOOGLE_PROJECT_ID"), "location": os.getenv("GOOGLE_LOCATION", "us-central1")},
    # you should set the context window to the max input tokens for the model
    context_window=int(os.getenv("CONTEXT_WINDOW", "200000")),
    max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
     vertexai=True,
)

        embed_model = GoogleGenAIEmbedding(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-005"),
        vertexai_config={"project": os.getenv("GOOGLE_PROJECT_ID"), "location": os.getenv("GOOGLE_LOCATION", "us-central1")},
        vertexai=True,
    )
        Settings.embed_model = embed_model
        Settings.llm = llm

        documents = SimpleDirectoryReader(temp_dir).load_data()

        index = VectorStoreIndex.from_documents(
        documents,
        transformations=[SentenceSplitter(chunk_size=int(os.getenv("CHUNK_SIZE", "1024")), chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "256")))],
    )
        splitter = SemanticSplitterNodeParser(buffer_size= 1, breakpoint_percentile_threshold=int(os.getenv("SEMANTIC_THRESHOLD", "95")), embed_model=embed_model)
        nodes = splitter.get_nodes_from_documents(documents)
        hyde = HyDEQueryTransform(include_original=True)

        query_engine = index.as_query_engine(similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "10")),
        )
        query_engine = TransformQueryEngine(query_engine, query_transform=hyde)

        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        # Generate session ID and create session
        files_content = b"".join([await file.read() for file in files]) if files else b""
        session_id = generate_session_id(files_content, user_related_data.user_query, user_related_data.num_of_questions)
        
        # Create initial session
        session = ProcessingSession(
            session_id=session_id,
            total_chunks=len(nodes),
            processed_chunks=0,
            all_mcqs=[],
            total_questions_generated=0,
            target_questions=user_related_data.num_of_questions,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Start processing with restart capability
        return await continue_processing_with_nodes(session, nodes, user_related_data, query_engine, background_tasks)

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}

async def continue_processing(session: ProcessingSession, background_tasks: BackgroundTasks):
    """Continue processing from saved session"""
    # Note: In a real implementation, you'd need to rebuild the nodes and query_engine
    # For now, return the current state
    return JSONResponse(content={
        "mcqs": session.all_mcqs, 
        "total_generated": session.total_questions_generated,
        "session_id": session.session_id,
        "processed_chunks": session.processed_chunks,
        "total_chunks": session.total_chunks,
        "status": "completed" if session.total_questions_generated >= session.target_questions else "incomplete"
    })

async def continue_processing_with_nodes(session: ProcessingSession, nodes, user_related_data, query_engine, background_tasks):
    """Process nodes with checkpoint saving"""
    try:
        questions_per_chunk = max(1, user_related_data.num_of_questions // len(nodes)) if len(nodes) > 0 else user_related_data.num_of_questions
        llm = Settings.llm
        
        # Process each node/chunk to generate questions continuously
        for i, node in enumerate(nodes):
            # Skip already processed chunks
            if i < session.processed_chunks:
                continue
                
            if session.total_questions_generated >= session.target_questions:
                break
                
            # Calculate remaining questions needed
            remaining_questions = session.target_questions - session.total_questions_generated
            questions_for_this_chunk = min(questions_per_chunk, remaining_questions)
            
            # Use node content directly for more targeted question generation
            node_content = node.get_content()
            
            # Create a query based on user query and node content
            if user_related_data.user_query:
                content = query_engine.query(f"{user_related_data.user_query} based on: {node_content[:int(os.getenv('NODE_CONTENT_LIMIT', '1000'))]}")
            else:
                content = node_content
            
            prompt = f"""
            Let's think step by step. 
            I need you to generate {questions_for_this_chunk} UPSC-style multiple choice questions based STRICTLY on the content I will provide.
            Each question should have 4 options (A, B, C, D), with only one correct answer.
            Indicate the correct answer after each question. The entire output MUST be in JSON format as an array of question objects.
            Do not include any introductory or explanatory text. Just the JSON array.

            The content is: {content}
            YOU ARE NOT SUPPOSED TO MAKE ANY ABSTRACT REFERENCE TO THE CONTENT LIKE CONTENT, TEXT, INFORMATION, DATA, OR ANY OTHER SIMILAR WORDS.
            You should not refer to the content in an abstract manner.
            IF YOU NEED TO REFER TO THE CONTENT, DO IT IN A DIRECT MANNER, PROVIDING THE ENTIRE CONTENT.
            You are supposed to use the content and the words within the node_content to create questions, you are not allowed to refer the content in abstract manner if need be you can give the entire content. Do not invent external information. Do not refer to the variable "node_content" itself in any form; only use the content within it, that it contains.
            
            Example of the required output format is:
            [
              {{
                "question": "What is the primary significance of the 'Gramophone' in the context of early 20th-century Indian society?",
                "options": {{
                  "A": "It facilitated the widespread dissemination of Western classical music, leading to a cultural shift.",
                  "B": "It played a crucial role in the development of nationalist sentiments by broadcasting patriotic songs and speeches.",
                  "C": "It primarily served as a tool for entertainment and leisure, with limited social or political impact.",
                  "D": "It was instrumental in the standardization of Indian languages through its use in radio broadcasts."
                }},
                "correct_answer": "B"
              }}
            ]
            
            If the query mentions GRE or GMAT, then use this format with 5 options (A-E) and focus on vocabulary/reasoning questions.
            """
            
            try:
                response = llm.complete(prompt)
                generated_content = response.text
                
                # Clean and parse the JSON response
                cleaned_json = rev_valid_json_string(generated_content)
                chunk_mcqs = json.loads(cleaned_json)
                
                # Ensure it's a list
                if isinstance(chunk_mcqs, dict):
                    chunk_mcqs = [chunk_mcqs]
                elif not isinstance(chunk_mcqs, list):
                    chunk_mcqs = []
                
                # Add questions from this chunk to our total collection
                for mcq in chunk_mcqs:
                    if session.total_questions_generated < session.target_questions:
                        session.all_mcqs.append(mcq)
                        session.total_questions_generated += 1
                        
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing JSON for chunk {i}: {e}")
            
            # Update session progress
            session.processed_chunks = i + 1
            session.last_updated = datetime.now()
            save_session(session)
            
            # Check for token limit or restart condition
            if session.total_questions_generated >= session.target_questions:
                break
        
        # If we still need more questions, try fallback generation
        if session.total_questions_generated < session.target_questions:
            remaining = session.target_questions - session.total_questions_generated
            content = query_engine.query(user_related_data.user_query) if user_related_data.user_query else "Generate general knowledge questions"
            
            prompt = f"""
            Generate {remaining} additional UPSC-style multiple choice questions based on the content: {content}
            Return as JSON array format as shown in previous examples.
            """
            
            try:
                response = llm.complete(prompt)
                generated_content = response.text
                
                cleaned_json = rev_valid_json_string(generated_content)
                additional_mcqs = json.loads(cleaned_json)
                
                if isinstance(additional_mcqs, dict):
                    additional_mcqs = [additional_mcqs]
                elif isinstance(additional_mcqs, list):
                    for mcq in additional_mcqs:
                        if session.total_questions_generated < session.target_questions:
                            session.all_mcqs.append(mcq)
                            session.total_questions_generated += 1
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing additional JSON: {e}")
            
            # Update final session state
            session.last_updated = datetime.now()
            save_session(session)

        return JSONResponse(content={
            "mcqs": session.all_mcqs, 
            "total_generated": len(session.all_mcqs),
            "session_id": session.session_id,
            "processed_chunks": session.processed_chunks,
            "total_chunks": session.total_chunks,
            "status": "completed" if session.total_questions_generated >= session.target_questions else "processing"
        })

    except RuntimeError as e:
        # Save session state before exiting due to token limit
        session.last_updated = datetime.now()
        save_session(session)
        
        return JSONResponse(content={
            "mcqs": session.all_mcqs,
            "total_generated": len(session.all_mcqs), 
            "session_id": session.session_id,
            "processed_chunks": session.processed_chunks,
            "total_chunks": session.total_chunks,
            "status": "interrupted",
            "message": "Processing interrupted due to token limit. Use resume endpoint to continue.",
            "error": str(e)
        })
    
    except Exception as e:
        session.last_updated = datetime.now()
        save_session(session)
        return {"error": str(e), "trace": traceback.format_exc(), "session_id": session.session_id}

@app.get("/session-status/{session_id}")
async def get_session_status(session_id: str):
    """Get the current status of a processing session"""
    session = load_session(session_id)
    if not session:
        return JSONResponse(content={"error": "Session not found or expired"}, status_code=404)
    
    return JSONResponse(content={
        "session_id": session.session_id,
        "total_chunks": session.total_chunks,
        "processed_chunks": session.processed_chunks,
        "total_questions_generated": session.total_questions_generated,
        "target_questions": session.target_questions,
        "progress_percentage": (session.processed_chunks / session.total_chunks * 100) if session.total_chunks > 0 else 0,
        "created_at": session.created_at.isoformat(),
        "last_updated": session.last_updated.isoformat(),
        "status": "completed" if session.total_questions_generated >= session.target_questions else "processing"
    })

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a processing session"""
    session_file = f"processing_sessions/{session_id}.pkl"
    if os.path.exists(session_file):
        os.remove(session_file)
        return JSONResponse(content={"message": "Session deleted successfully"})
    else:
        return JSONResponse(content={"error": "Session not found"}, status_code=404)
