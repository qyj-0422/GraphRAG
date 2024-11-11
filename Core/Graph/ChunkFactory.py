

from Core.Common.Constants import Default_text_separator
import tiktoken
from pydantic import BaseModel
from typing import List, Optional, Union, Literal, Any
from Core.Common.Utils import mdhash_id

# Used for chunking the text, so we place it in here.  
class SeparatorSplitter:
    def __init__(
        self,
        separators: Optional[List[List[int]]] = None,
        keep_separator: Union[bool, Literal["start", "end"]] = "end",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: callable = len,
    ):
        self._separators = separators or []
        self._keep_separator = keep_separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function

    def split_tokens(self, tokens: List[int]) -> List[List[int]]:
        splits = self._split_tokens_with_separators(tokens)
        return self._merge_splits(splits)

    def _split_tokens_with_separators(self, tokens: List[int]) -> List[List[int]]:
        splits = []
        current_split = []
        i = 0
        while i < len(tokens):
            separator_found = False
            for separator in self._separators:
                if tokens[i:i+len(separator)] == separator:
                    if self._keep_separator in [True, "end"]:
                        current_split.extend(separator)
                    if current_split:
                        splits.append(current_split)
                        current_split = []
                    if self._keep_separator == "start":
                        current_split.extend(separator)
                    i += len(separator)
                    separator_found = True
                    break
            if not separator_found:
                current_split.append(tokens[i])
                i += 1
        if current_split:
            splits.append(current_split)
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[List[int]]) -> List[List[int]]:
        if not splits:
            return []

        merged_splits = []
        current_chunk = []

        for split in splits:
            if not current_chunk:
                current_chunk = split
            elif self._length_function(current_chunk) + self._length_function(split) <= self._chunk_size:
                current_chunk.extend(split)
            else:
                merged_splits.append(current_chunk)
                current_chunk = split

        if current_chunk:
            merged_splits.append(current_chunk)

        if len(merged_splits) == 1 and self._length_function(merged_splits[0]) > self._chunk_size:
            return self._split_chunk(merged_splits[0])

        if self._chunk_overlap > 0:
            return self._enforce_overlap(merged_splits)
        
        return merged_splits

    def _split_chunk(self, chunk: List[int]) -> List[List[int]]:
        result = []
        for i in range(0, len(chunk), self._chunk_size - self._chunk_overlap):
            new_chunk = chunk[i:i + self._chunk_size]
            if len(new_chunk) > self._chunk_overlap:  # 只有当 chunk 长度大于 overlap 时才添加
                result.append(new_chunk)
        return result

    def _enforce_overlap(self, chunks: List[List[int]]) -> List[List[int]]:
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                overlap = chunks[i-1][-self._chunk_overlap:]
                new_chunk = overlap + chunk
                if self._length_function(new_chunk) > self._chunk_size:
                    new_chunk = new_chunk[:self._chunk_size]
                result.append(new_chunk)
        return result


class ChunkingFactory(BaseModel):
    
    chunk_methods:dict = {}
    def register_chunking_method(
        self,
        method_name: str,
        method_func=None  # can be any classes or functions
    ):
        if self.has_chunk_method(method_name):
            return

        self.chunk_methods[method_name] = method_func
      

    def has_chunk_method(self, key: str) -> Any:
        return key in self.chunk_methods

    def get_method(self, key) -> Any:
        return self.chunk_methods.get(key)
    

# Registry instance
Chunking_REGISTRY = ChunkingFactory()


def register_chunking_method(method_name):
    """ Register a new chunking method
    
    This is a decorator that can be used to register a new chunking method.
    The method will be stored in the self.methods dictionary.
    
    Parameters
    ----------
    method_name: str
        The name of the chunking method.
    """
    def decorator(func):
        """ Register a new chunking method """
        Chunking_REGISTRY.register_chunking_method(method_name, func)
    
    return decorator



@register_chunking_method("chunking_by_token_size")
async def chunking_by_token_size(tokens_list: list[list[int]], doc_keys, tiktoken_model, overlap_token_size=128, max_token_size=1024):
    """
    Chunking by token size.

    This method will split the tokens list into chunks given a max token size.

    :return: A list of chunks.
    :rtype: list
    """

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results

@register_chunking_method("chunking_by_seperators")
async def chunking_by_seperators( tokens_list: list[list[int]], doc_keys, tiktoken_model, overlap_token_size=128, max_token_size=1024):
        """
        Chunking by separators.

        This method will split the tokens list into chunks given a list of separators.

        :param overlap_token_size: the number of tokens to overlap between chunks
        :type overlap_token_size: int
        :param max_token_size: the maximum number of tokens in a chunk
        :type max_token_size: int
        :return: A list of chunks.
        :rtype: list
        """
        # TODO: add the logic to set the default separators
        splitter = SeparatorSplitter(
            separators=[
            tiktoken_model.encode(sep) for sep in Default_text_separator],
            chunk_size=max_token_size,
            chunk_overlap=overlap_token_size,
        )
        results = []
        for index, tokens in enumerate(tokens_list):
            chunk_token = splitter.split_tokens(tokens)
            lengths = [len(c) for c in chunk_token]

            # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
            chunk_token = tiktoken_model.decode_batch(chunk_token)
            for i, chunk in enumerate(chunk_token):

                results.append(
                    {
                        "tokens": lengths[i],
                        "content": chunk.strip(),
                        "chunk_order_index": i,
                        "full_doc_id": doc_keys[index],
                    }
                )

        return results


def create_chunk_method(method_name):

    chunking_method = Chunking_REGISTRY.get_method(method_name)
    return chunking_method


async def get_chunks(new_docs, chunk_method_name, token_model, **chunk_func_params):

    kv_chunks = {}

    new_docs_list = list(new_docs.items())
    docs = [new_doc[1]["content"] for new_doc in new_docs_list]
    doc_keys = [new_doc[0] for new_doc in new_docs_list]

    tokens = token_model.encode_batch(docs, num_threads=16)


    chunk_func = create_chunk_method(chunk_method_name)

    chunks = await chunk_func(
        tokens, doc_keys=doc_keys, tiktoken_model=token_model, **chunk_func_params
    )

    for chunk in chunks:
        kv_chunks.update(
            {mdhash_id(chunk["content"], prefix="chunk-"): chunk}
        )

    return kv_chunks