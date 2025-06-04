import re
import json
import tiktoken
from pathlib import Path
from collections import defaultdict
import logging
from typing import List, Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.language_models.base import BaseLanguageModel

from deeprecall.services.celery_app import celery_app
from deeprecall.services.providers.embeding import create_embedding
from deeprecall.services.providers.vectorstore import get_vectorstore
from deeprecall.services.providers.llm import create_llm


# Configure logging
logging.basicConfig(
    filename="extract.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def extract_relevant_documents(
    query: str,
    collection_name: str,
    vectorstore_provider: str,
    embedding_provider: str,
    embedding_key: str,
    embedding_model: str = None,
    embedding_url: str = None,
    embedding_provider_id: str = None,
    k: int = 4,
) -> Dict[str, Any]:
    """
    Extract relevant documents from Milvus vector store for LLM context.

    Args:
        self: Celery task instance
        query: Search query text
        collection_name: Name of the Milvus collection to search
        embedding_provider: Name of the embedding provider to use
        k: Number of documents to retrieve (default: 4)

    Returns:
        Dict containing search results and metadata
    """
    try:
        # Validate required parameters
        if not all([query, collection_name, embedding_provider]):
            error_msg = "Missing required parameters: query, collection_name, and embedding_provider"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

        # Get embedding provider
        embedding_provider = (
            embedding_provider.lower().replace(" ", "_").replace("-", "_")
        )
        # Create embedding engine using the same pattern as embed task
        kwargs = {"provider_name": embedding_provider, "key": embedding_key}
        if embedding_model is not None:
            kwargs["model"] = embedding_model
        if embedding_url is not None:
            kwargs["url"] = embedding_url
        if embedding_provider_id is not None:
            kwargs["provider_id"] = embedding_provider_id
        embed_engine = create_embedding(**kwargs)

        if embed_engine is None:
            error_msg = f"Unsupported embedding provider: {embedding_provider}"
            logging.error(error_msg)
            return {"success": False, "error": error_msg}

        # Use vectorstore provider factory pattern like embed task
        vector_store = get_vectorstore(
            provider_name=vectorstore_provider,
            collection_name=collection_name,
            embedding_function=embed_engine,
            init_collection=False,
        )

        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)

        # Format results for LLM context
        formatted_results = [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source": doc.metadata.get("source", "unknown"),
            }
            for doc in results
        ]

        return {
            "success": True,
            "query": query,
            "collection": collection_name,
            "results": formatted_results,
            "count": len(formatted_results),
        }

    except Exception as e:
        error_msg = f"Critical error in extract_relevant_documents: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg}


@celery_app.task(bind=True)
def preprocess(
    self,
    job_id: str,
    input_folder_path: str,
    output_folder_path: str,
    vectorstore_provider: str,
    embedding_provider: str,
    embedding_model: str = None,
    embedding_key: str = None,
    embedding_url: str = None,
    embedding_provider_id: str = None,
    llm_provider: str = None,
    llm_model: str = None,
    llm_key: str = None,
    llm_url: str = None,
    llm_provider_id: str = None,
    context_size: int = 8192,
):
    try:
        # Convert to Path objects immediately
        input_path = Path(input_folder_path)
        output_path = Path(output_folder_path) / job_id
        doc_chunk_path = output_path / "doc_chunks"
        intermediate_path = output_path / "intermediate"
        results_path = output_path / "results"

        # Create output directories if they don't exist
        doc_chunk_path.mkdir(parents=True, exist_ok=True)
        intermediate_path.mkdir(parents=True, exist_ok=True)
        results_path.mkdir(parents=True, exist_ok=True)

        # Validate input directory
        if not input_path.exists():
            error_msg = f"Input folder path {input_path} does not exist"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

        # Validate doc_chunks directory
        if not any(doc_chunk_path.iterdir()):
            error_msg = f"Document chunks directory {doc_chunk_path} is empty"
            logging.error(error_msg)
            self.update_state(state="FAILURE", meta={"error": error_msg})
            return False

        # Step 1: Process file chunks
        chunk_files = list(doc_chunk_path.glob("*_chunk_*"))
        grouped = defaultdict(list)

        for file in chunk_files:
            if match := re.match(r"(.*?)_chunk_(\d+)\.txt$", file.name):
                doc_name = match.group(1)
                grouped[doc_name].append(file)

        # Create LLM instance
        llm = create_llm(
            provider_name=llm_provider,
            model=llm_model,
            key=llm_key,
            url=llm_url,
            provider_id=llm_provider_id,
        )

        # Step 2: Generate summary and questions
        for doc_name, files in grouped.items():
            questions = []
            summaries = []

            # Sort files numerically
            files.sort(
                key=lambda x: int(re.search(r"_chunk_(\d+)\.txt$", x.name).group(1))
            )

            for file in files:
                content = file.read_text(encoding="utf-8")
                summaries.append(generate_summary(content, llm))
                questions.extend(generate_questions(content, llm))

            # Write intermediate files
            (intermediate_path / f"{doc_name}_summaries.txt").write_text(
                "\n".join(summaries), encoding="utf-8"
            )
            (intermediate_path / f"{doc_name}_questions.txt").write_text(
                "\n".join(questions), encoding="utf-8"
            )

        # Step 3: Process intermediate files
        for file_path in intermediate_path.iterdir():
            if not file_path.name.endswith("_summaries.txt"):
                continue

            doc_name = file_path.stem.replace("_summaries", "")
            questions_path = intermediate_path / f"{doc_name}_questions.txt"

            if not questions_path.exists():
                logging.warning(f"Questions file missing for {doc_name}")
                continue

            summaries = file_path.read_text(encoding="utf-8")
            questions = questions_path.read_text(encoding="utf-8").splitlines()

            # Process summaries
            condensed_summary, was_condensed = process_summaries(
                summaries, questions, llm, context_size
            )

            # Process questions
            merged_questions = merge_questions(questions, condensed_summary, llm)
            expanded_questions = expand_questions(
                merged_questions, condensed_summary, llm
            )

            # Step 4: Generate QA pairs
            generate_qa_pairs(
                doc_name,
                expanded_questions,
                llm,
                results_path,
                job_id,
                vectorstore_provider,
                embedding_provider,
                embedding_key,
                embedding_model=embedding_model,
                embedding_url=embedding_url,
                embedding_provider_id=embedding_provider_id,
                k=4,
            )

    except Exception as e:
        logging.error(f"Critical error in preprocess: {str(e)}", exc_info=True)
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return False


# Token counting function
def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def generate_summary(content: str, llm: BaseLanguageModel) -> str:
    """Generate a one-sentence summary using the specified LLM provider."""
    prompt = PromptTemplate(
        template="Write a ONE-SENTENCE summary of the following text:\n\n{content}",
        input_variables=["content"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"content": content}).strip()


def generate_questions(content: str, llm: BaseLanguageModel) -> List[str]:
    """Generate questions using the specified LLM provider."""
    prompt = PromptTemplate(
        template="Generate 3-5 potential questions users might ask about this content:\n{content}\n\nQuestions:\n",
        input_variables=["content"],
    )
    chain = prompt | llm | StrOutputParser()
    output = chain.invoke({"content": content}).strip()
    return [q.strip() for q in output.split("\n") if q.strip()]


def process_summaries(
    summaries: str,
    questions: List[str],
    llm: BaseLanguageModel,
    max_context_tokens: int,
):
    """Condense summaries if combined with questions exceeds context length"""
    # Calculate token counts
    questions_text = "\n".join(questions)
    combined_text = f"{summaries}\n\n{questions_text}"

    if count_tokens(combined_text) <= max_context_tokens:
        return summaries, False

    # Condense summaries
    prompt = PromptTemplate(
        template="Condense these document summaries while preserving key information:\n{summaries}\n\nCondensed Summary:",
        input_variables=["summaries"],
    )
    chain = prompt | llm | StrOutputParser()
    condensed = chain.invoke({"summaries": summaries}).strip()

    # Recursively condense if needed
    if count_tokens(f"{condensed}\n\n{questions_text}") > max_context_tokens:
        return process_summaries(condensed, questions)

    return condensed, True


def merge_questions(questions: str, summary_context: str, llm: BaseLanguageModel):
    """Merge similar questions using summary as context"""
    prompt_template = """Using this document summary as context, merge similar questions:
    Summary: {context}
    
    Questions:
    {questions}
    
    Instructions:
    1. Remove duplicates
    2. Merge similar questions
    3. Keep only distinct questions
    4. Output one question per line
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "questions"]
    )
    chain = prompt | llm | StrOutputParser()
    output = chain.invoke(
        {"context": summary_context, "questions": "\n".join(questions)}
    ).strip()
    return [q.strip() for q in output.split("\n") if q.strip()]


def expand_questions(questions: str, summaries: str, llm: BaseLanguageModel):
    prompt = PromptTemplate(
        template="Generate additional related questions based on these summaries:\n{summaries}\n\nExisting Questions:\n{questions}\n\nNew Questions (one per line):",
        input_variables=["summaries", "questions"],
    )
    chain = prompt | llm | StrOutputParser()
    output = chain.invoke(
        {"summaries": summaries, "questions": "\n".join(questions)}
    ).strip()
    return [q.strip() for q in output.split("\n") if q.strip()]


def generate_qa_pairs(
    doc_name: str,
    questions: List[str],
    llm: BaseLanguageModel,
    results_dir: Path,
    collection_name: str,
    vectorstore_provider: str,
    embedding_provider: str,
    embedding_key: str,
    embedding_model: str = None,
    embedding_url: str = None,
    embedding_provider_id: str = None,
    k: int = 4,
):
    for i, question in enumerate(questions):
        try:
            # Retrieve relevant documents
            result = extract_relevant_documents(
                query=question,
                collection_name=collection_name,
                embedding_key=embedding_key,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                embedding_provider_id=embedding_provider_id,
                embedding_url=embedding_url,
                vectorstore_provider=vectorstore_provider,
                k=k,
            )

            if not result["success"] or not result["results"]:
                continue

            # Extract context directly
            context = "\n".join(result["results"])

            # Generate answer
            answer = generate_answer(question, context)

            # Save QA pair
            save_qa_pair(doc_name, i, question, answer)

        except Exception as e:
            print(f"Error processing question {i} for {doc_name}: {str(e)}")


def generate_answer(question: str, context: str, llm: BaseLanguageModel) -> str:
    prompt = PromptTemplate(
        template="Answer this question based ONLY on the context:\nQuestion: {question}\nContext: {context}\nAnswer:",
        input_variables=["question", "context"],
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": context}).strip()


def save_qa_pair(
    doc_name: str, idx: int, question: str, answer: str, results_dir: Path
):
    # Create separate message objects
    user_msg = {"role": "user", "content": question}
    assistant_msg = {"role": "assistant", "content": answer}

    filename = f"{doc_name}_qa_{idx}.json"
    file_path = results_dir / filename
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(user_msg, f)
        f.write("\n")  # Newline separator
        json.dump(assistant_msg, f)
