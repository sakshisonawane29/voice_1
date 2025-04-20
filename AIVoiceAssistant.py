from qdrant_client import QdrantClient, models
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import Document
from llama_index.core.indices.list import ListIndex
import unittest.mock as mock
import os
import json
import requests
import time

import warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self):
        # Set your Hugging Face API key
        self._hf_api_key = "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"
        
        # Validate API key before proceeding
        self._validate_api_key()
        
        # Set environment variable after validation
        os.environ["HUGGINGFACE_API_KEY"] = self._hf_api_key
        
        # Connect to Qdrant vector database (cloud instance)
        self._qdrant_url = "https://e1d98560-beac-4ca6-9f5c-8761a291230d.us-east4-0.gcp.cloud.qdrant.io:6333"
        self._qdrant_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C9_TgqQ03AEbF56k7YcOmuXx6X_Rlfr00PR2jY2BUQk"
        
        # Initialize Qdrant client with cloud credentials and fallback options
        try:
            print("Attempting to connect to cloud Qdrant instance...")
            self._client = QdrantClient(
                url=self._qdrant_url,
                api_key=self._qdrant_api_key,
                prefer_grpc=False  # Use HTTP for cloud instance
            )
            print("Successfully connected to cloud Qdrant instance")
        except Exception as e:
            print(f"Error connecting to cloud Qdrant: {str(e)}")
            print("Falling back to local Qdrant instance...")
            try:
                self._qdrant_url = "http://localhost:6333"
                self._client = QdrantClient(
                    url=self._qdrant_url,
                    prefer_grpc=False  # Use HTTP for more reliable connection
                )
                print("Successfully connected to local Qdrant instance")
            except Exception as local_e:
                print(f"Error connecting to local Qdrant: {str(local_e)}")
                print("WARNING: No Qdrant connection available. RAG functionality will be limited.")
                # Creating a minimal client that will be replaced once connection is available
                self._client = None
        
        self._collection_name = "taj_hotel_info"
        
        # Use smaller, faster embedding model
        self._embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",  # Smaller, faster model
            cache_folder="./embedding_cache"  # Cache embeddings for repeated queries
        )
        
        # Initialize Mistral LLM via Hugging Face API with optimization parameters
        self._llm = HuggingFaceInferenceAPI(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            token=self._hf_api_key,
            timeout=60,  # Reduced timeout
            model_kwargs={
                "max_new_tokens": 256,  # Limit response length
                "temperature": 0.5,     # Lower temperature for more focused responses
                "top_p": 0.9,           # Use nucleus sampling for speed
                "repetition_penalty": 1.1  # Slight penalty to avoid repetitions
            }
        )
        
        # Set up optimized service context
        from llama_index.core.settings import Settings
        
        Settings.embed_model = self._embed_model
        Settings.llm = self._llm
        Settings.chunk_size = 256  # Smaller chunk size for faster processing
        Settings.chunk_overlap = 20  # Minimal overlap
        
        self._service_context = ServiceContext.from_defaults(
            llm=self._llm,
            embed_model=self._embed_model,
            chunk_size=256,
            chunk_overlap=20
        )
        
        self._index = None
        self._chat_memory = None
        self._prompt = """You are an AI receptionist for the historic Taj Mahal Palace Hotel in Mumbai, India. Be professional, courteous, and helpful.
        Provide accurate information about our hotel's rich history, luxury services, iconic restaurants, accommodation options, and nearby tourist attractions based on the knowledge base.
        When guests ask about places to visit or explore around the hotel, provide details on distance, transportation options, and any special recommendations.
        Always start your response with 'Taj Mahal Palace Mumbai Assistant:'
        Keep your responses concise and focused on the most relevant information.
        If you do not have the information requested in your knowledge base, clearly state that you don't have that information and advise the guest to contact the hotel's reception desk at +91-22-6665-3366 or email at tmhbc.bom@tajhotels.com for more details."""
        
        # Initialize Qdrant collection
        self._init_qdrant_collection()
        # Try to load existing index
        self._load_or_create_kb()
        self._create_chat_engine()

    def _validate_api_key(self):
        """Validate the Hugging Face API key"""
        headers = {"Authorization": f"Bearer {self._hf_api_key}"}
        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        try:
            response = requests.post(url, headers=headers, json={"inputs": "test"})
            if response.status_code == 401:
                print(f"API Key validation failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                raise ValueError("Invalid Hugging Face API key")
        except Exception as e:
            print(f"Error during API key validation: {str(e)}")
            raise

    def _init_qdrant_collection(self):
        """Initialize or create Qdrant collection with error handling and retries"""
        # If client is None, we can't initialize collection
        if self._client is None:
            print("Qdrant client is not available. Collection initialization skipped.")
            return
            
        # Try up to 3 times to initialize collection
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt+1}/{max_retries} to initialize Qdrant collection...")
                
                # Check if collection exists
                collections = self._client.get_collections().collections
                exists = any(col.name == self._collection_name for col in collections)
                
                if exists:
                    print(f"Collection '{self._collection_name}' already exists")
                    # If collection exists, update optimization parameters if needed
                    try:
                        self._client.update_collection(
                            collection_name=self._collection_name,
                            optimizer_config=models.OptimizersConfigDiff(
                                memmap_threshold=1000
                            )
                        )
                        print(f"Updated collection '{self._collection_name}' with optimization parameters")
                    except Exception as e:
                        print(f"Warning: Could not update collection parameters: {str(e)}")
                    
                    return  # Collection exists, no need to create it
                else:
                    # Create new collection with optimized configuration
                    print(f"Creating new collection: '{self._collection_name}'")
                    self._client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=models.VectorParams(
                            size=384,  # Dimension for embedding model
                            distance=models.Distance.COSINE
                        ),
                        # Adding optimized parameters for faster search
                        hnsw_config=models.HnswConfigDiff(
                            m=16,                # Number of edges per node
                            ef_construct=100,    # Size of the dynamic candidate list during indexing
                            full_scan_threshold=10000  # Threshold for using HNSW index vs full scan
                        ),
                        optimizers_config=models.OptimizersConfigDiff(
                            memmap_threshold=1000  # Threshold for using memory mapping
                        )
                    )
                    print(f"Successfully created collection '{self._collection_name}'")
                    return  # Successfully created collection
                
            except Exception as e:
                print(f"Error during collection initialization (attempt {attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("All retry attempts failed. Collection initialization unsuccessful.")
                    print("WARNING: Vector search functionality will be limited.")
                    # Create empty placeholder to avoid None errors later
                    import mock
                    self._client = mock.Mock()
                    return

    def _load_or_create_kb(self):
        """Load existing knowledge base or create new one with error handling"""
        # If client was mocked due to connection issues, use alternative approach
        if self._client is None or isinstance(self._client, mock.Mock):
            print("Using alternative knowledge base approach (no vector store)")
            try:
                # Create a simple in-memory index without vector store
                current_dir = os.path.dirname(os.path.abspath(__file__))
                hotel_txt_path = os.path.join(current_dir, "hotel.txt")
                
                if not os.path.exists(hotel_txt_path):
                    raise FileNotFoundError(f"Hotel knowledge base file not found at: {hotel_txt_path}")
                
                # Read hotel.txt content directly
                with open(hotel_txt_path, 'r', encoding='utf-8') as f:
                    self._hotel_content = f.read()
                    
                print("Loaded hotel information into memory")
                
                # Create a simple list-based index
                from llama_index.core import Document
                from llama_index.core.indices.list import ListIndex
                
                document = Document(text=self._hotel_content)
                self._index = ListIndex.from_documents(
                    [document], 
                    service_context=self._service_context
                )
                print("Created in-memory list index as fallback")
                
            except Exception as e:
                print(f"Error creating alternative knowledge base: {str(e)}")
                # Create empty index to avoid None errors
                document = Document(text="Basic Taj Mahal Palace Hotel information")
                self._index = ListIndex.from_documents(
                    [document], 
                    service_context=self._service_context
                )
                return
                
        # Normal vector store approach
        try:
            print("Setting up vector store knowledge base...")
            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=self._collection_name
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Check if collection has documents
            try:
                collection_info = self._client.get_collection(self._collection_name)
                if collection_info.points_count > 0:
                    print(f"Loading existing knowledge base with {collection_info.points_count} points")
                    self._index = VectorStoreIndex.from_vector_store(
                        vector_store,
                        service_context=self._service_context
                    )
                    print("Successfully loaded existing knowledge base")
                    return
            except Exception as e:
                print(f"Error checking collection: {str(e)}")
            
            # If we got here, we need to create a new knowledge base
            print("No existing knowledge base found, creating new one...")
            self._create_new_kb(storage_context)
                
        except Exception as e:
            print(f"Error in vector knowledge base setup: {str(e)}")
            print("Falling back to simpler in-memory index...")
            
            # Create an in-memory index as fallback
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                hotel_txt_path = os.path.join(current_dir, "hotel.txt")
                
                if not os.path.exists(hotel_txt_path):
                    raise FileNotFoundError(f"Hotel knowledge base file not found at: {hotel_txt_path}")
                
                from llama_index.core import Document
                from llama_index.core.indices.list import ListIndex
                
                with open(hotel_txt_path, 'r', encoding='utf-8') as f:
                    hotel_content = f.read()
                
                document = Document(text=hotel_content)
                self._index = ListIndex.from_documents(
                    [document], 
                    service_context=self._service_context
                )
                print("Created in-memory list index as fallback")
            except Exception as fallback_e:
                print(f"Error creating fallback index: {str(fallback_e)}")
                # Last resort - create minimal working index
                document = Document(text="Basic Taj Mahal Palace Hotel information")
                self._index = ListIndex.from_documents(
                    [document], 
                    service_context=self._service_context
                )

    def _create_new_kb(self, storage_context):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            hotel_txt_path = os.path.join(current_dir, "hotel.txt")
            
            if not os.path.exists(hotel_txt_path):
                raise FileNotFoundError(f"Hotel knowledge base file not found at: {hotel_txt_path}")
            
            reader = SimpleDirectoryReader(input_files=[hotel_txt_path])
            documents = reader.load_data()
            
            self._index = VectorStoreIndex.from_documents(
                documents,
                service_context=self._service_context,
                storage_context=storage_context
            )
            
        except Exception as e:
            print(f"Error creating knowledge base: {str(e)}")
            raise

    def _create_chat_engine(self):
        """Create an appropriate chat engine based on available index type"""
        if self._index is None:
            raise ValueError("Index not created. Please ensure knowledge base creation was successful.")
        
        # Initialize chat memory with a smaller token limit for faster processing
        self._chat_memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
        
        try:
            print("Creating optimized query engine...")
            # Check the type of index to determine the appropriate engine creation
            if isinstance(self._index, VectorStoreIndex):
                # Vector store index - use retriever based engine
                from llama_index.core.query_engine import RetrieverQueryEngine
                from llama_index.core.retrievers import VectorIndexRetriever
                from llama_index.core.response_synthesizers import CompactAndRefine
                
                # Configure a retriever with optimized parameters
                retriever = VectorIndexRetriever(
                    index=self._index,
                    similarity_top_k=2,  # Reduce top_k for faster retrieval
                    search_kwargs={"search_type": "mmr", "mmr_diversity_bias": 0.3}  # Use maximal marginal relevance for diverse results
                )
                
                # Create a response synthesizer with optimized parameters
                response_synthesizer = CompactAndRefine(
                    service_context=self._service_context,
                    streaming=False,
                    verbose=False,
                )
                
                # Create query engine with the optimized components
                self._chat_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    response_synthesizer=response_synthesizer,
                    service_context=self._service_context
                )
                print("Created optimized retrieval-based query engine")
            else:
                # Default to basic query engine for other index types (like ListIndex)
                self._chat_engine = self._index.as_query_engine(
                    response_mode="compact",
                    service_context=self._service_context
                )
                print("Created basic query engine for non-vector index")
                
        except Exception as e:
            print(f"Error creating optimized query engine: {str(e)}")
            print("Falling back to simple query engine...")
            
            # Create a simple query engine as fallback
            try:
                self._chat_engine = self._index.as_query_engine(
                    response_mode="compact", 
                    service_context=self._service_context
                )
                print("Created fallback query engine")
            except Exception as fallback_e:
                print(f"Error creating fallback query engine: {str(fallback_e)}")
                # Create a function-based "pseudo-engine" as last resort
                def simple_query_function(query_str):
                    class SimpleResponse:
                        def __init__(self, text):
                            self.response = text
                        def __str__(self):
                            return self.response
                    
                    return SimpleResponse(
                        f"Taj Mahal Palace Mumbai Assistant: I have information about our hotel, but my search capabilities are limited at the moment. "
                        f"Please ask a specific question about our rooms, services, or nearby attractions."
                    )
                
                self._chat_engine = mock.Mock()
                self._chat_engine.query = simple_query_function
                print("Created emergency simple response function")

    def interact_with_llm(self, customer_query):
        try:
            # Check query length and simplify if too long
            if len(customer_query) > 100:
                customer_query = customer_query[:100] + "..."
            
            # Check if query is in cache
            # Simple in-memory cache implementation
            if not hasattr(self, '_query_cache'):
                self._query_cache = {}
            
            # Create cache key based on simplified query
            import hashlib
            cache_key = hashlib.md5(customer_query.lower().encode()).hexdigest()
            
            # Check if we have a cached response
            if cache_key in self._query_cache:
                print("Cache hit! Using cached response.")
                return self._query_cache[cache_key]
            
            # Process query to identify category for optimized handling
            query_lower = customer_query.lower()
            is_room_query = any(keyword in query_lower for keyword in ["room", "suite", "stay", "accommodation", "book"])
            is_restaurant_query = any(keyword in query_lower for keyword in ["restaurant", "dining", "food", "eat", "meal", "cuisine"])
            is_nearby_query = any(keyword in query_lower for keyword in ["nearby", "visit", "attraction", "place", "explore", "see", "tour"])
            is_transport_query = any(keyword in query_lower for keyword in ["transport", "taxi", "car", "travel", "ferry", "get to"])
            
            # Check for out-of-scope topics immediately
            out_of_scope_topics = [
                "stock market", "political", "election", "nuclear", "vaccine", 
                "pandemic", "controversial", "religious", "government", 
                "terrorist", "conspiracy", "president", "prime minister", 
                "war", "protest", "scandal"
            ]
            
            # If query is clearly out of scope, return direct contact information
            if any(topic in query_lower for topic in out_of_scope_topics):
                print(f"Query contains out-of-scope topic: '{customer_query}'")
                return "Taj Mahal Palace Mumbai Assistant: I apologize, but that topic is outside the scope of my hotel information. For any questions not related to our hotel services, facilities, or local attractions, please contact our concierge desk at +91-22-6665-3366 or email at tmhbc.bom@tajhotels.com for personalized assistance."
            
            # Optimize prompt based on query category
            if is_room_query:
                context = "Focus on room details, rates, and amenities."
            elif is_restaurant_query:
                context = "Focus on restaurant information, cuisines, and opening hours."
            elif is_nearby_query:
                context = "Focus on nearby attractions, distances, and visit durations."
            elif is_transport_query:
                context = "Focus on transportation options, costs, and arrangements."
            else:
                context = "Provide a brief, general response."
            
            # Compact formatted query for faster processing
            formatted_query = f"""
            Based on Taj Mahal Palace Hotel information, answer: {customer_query}
            {context} Be professional and concise.
            If you do not have the information in your knowledge base, politely state that you don't have that information and advise the guest to contact the hotel's reception desk at +91-22-6665-3366 for more details.
            """
            
            # Get response from query engine
            start_time = time.time()
            response = self._chat_engine.query(formatted_query)
            end_time = time.time()
            print(f"Query processing time: {end_time - start_time:.2f} seconds")
            
            # Format response
            response_text = str(response)
            
            # Use the dedicated method to check if response indicates query is outside knowledge base
            if self._is_out_of_knowledge_base(customer_query, response_text):
                # Enhance response with contact information if it's not already included
                if "+91-22-6665-3366" not in response_text:
                    response_text = f"{response_text.rstrip()} For more information on this specific query, please contact our reception desk at +91-22-6665-3366 or email at tmhbc.bom@tajhotels.com."
            
            # Ensure the response has the proper prefix
            if not response_text.startswith("Taj Mahal Palace Mumbai Assistant:"):
                response_text = "Taj Mahal Palace Mumbai Assistant: " + response_text
            
            # Cache the response for future use
            self._query_cache[cache_key] = response_text
            # Limit cache size to prevent memory issues
            if len(self._query_cache) > 100:
                # Remove oldest entry
                self._query_cache.pop(next(iter(self._query_cache)))
            
            return response_text
            
        except Exception as e:
            print(f"Error during LLM interaction: {str(e)}")
            return "Taj Mahal Palace Mumbai Assistant: I apologize, but I'm having trouble processing your request at the moment. Please contact our reception desk at +91-22-6665-3366 or email at tmhbc.bom@tajhotels.com for immediate assistance."

    def reset_chat(self):
        """Reset the chat engine with fresh memory"""
        self._create_chat_engine()

    def _is_out_of_knowledge_base(self, query, response_text):
        """
        Determines if a response indicates the query is outside the knowledge base.
        
        Args:
            query (str): The original query from the user
            response_text (str): The generated response text
            
        Returns:
            bool: True if the query appears to be outside the knowledge base
        """
        # Check for explicit uncertainty phrases
        uncertainty_phrases = [
            "i don't have that information",
            "i don't have specific information",
            "i'm not sure",
            "i don't know",
            "i cannot provide",
            "i do not have",
            "no information about",
            "i don't have details",
            "not in my knowledge base",
            "i'm unable to provide",
            "i apologize"
        ]
        
        # Check for hallucination indicators (very short responses, very generic answers)
        hallucination_indicators = [
            len(response_text) < 80,  # Very short responses often indicate lack of knowledge
            "please ask" in response_text.lower() and "specific" in response_text.lower(),
            "would you like" in response_text.lower() and "information" in response_text.lower(),
            "I'd be happy to help" in response_text and len(response_text) < 150
        ]
        
        # Check if these are topics typically outside hotel knowledge
        out_of_scope_topics = [
            "stock market",
            "political",
            "election",
            "nuclear",
            "vaccine",
            "pandemic",
            "controversial",
            "religious",
            "government",
            "terrorist",
            "conspiracy",
            "president",
            "prime minister",
            "war",
            "protest",
            "scandal"
        ]
        
        # Detect if query contains out-of-scope topics
        contains_out_of_scope = any(topic in query.lower() for topic in out_of_scope_topics)
        
        # Detect uncertainty in response
        shows_uncertainty = any(phrase in response_text.lower() for phrase in uncertainty_phrases)
        
        # Detect hallucination indicators
        shows_hallucination = any(hallucination_indicators)
        
        # Determine if likely outside knowledge base
        is_outside_kb = shows_uncertainty or (shows_hallucination and contains_out_of_scope)
        
        if is_outside_kb:
            print(f"Detected query outside knowledge base: '{query}'")
            
        return is_outside_kb
