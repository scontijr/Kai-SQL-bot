@@
-from langchain.embeddings.openai import OpenAIEmbeddings
-from langchain.vectorstores import FAISS
-from langchain.schema import Document
+from langchain.embeddings.openai import OpenAIEmbeddings
+from langchain.vectorstores import FAISS
+from langchain.schema import Document
+import os

-# existing few_shot_docs = [...] remains the same

-# OLD: this ran at import time and would crash the app if OpenAI was unreachable or misconfigured
-embeddings = OpenAIEmbeddings()  # implicit default may be ada-002 on older LangChain
-vector_db = FAISS.from_documents(few_shot_docs, embeddings)

+# NEW: pick a current model and defer building the index
+EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
+
+def _build_embeddings():
+    # For langchain==0.0.320 the class lives under langchain.embeddings.openai.OpenAIEmbeddings
+    # It accepts a `model` kwarg for the embeddings model.
+    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
+
+def build_few_shot_faiss():
+    """Build FAISS index for few-shot examples lazily."""
+    embeds = _build_embeddings()
+    return FAISS.from_documents(few_shot_docs, embeds)
+
+# Expose a callable instead of a prebuilt object
+def get_few_shot_vector_db():
+    return build_few_shot_faiss()
