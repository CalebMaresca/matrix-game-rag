from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from typing import Optional, Type

import logging
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, model_validator

WIKIPEDIA_MAX_QUERY_LENGTH = 300

class WikipediaAPIWrapper(BaseModel):
    """Wrapper around WikipediaAPI.

    To use, you should have the ``wikipedia`` python package installed.
    This wrapper will use the Wikipedia API to conduct searches and
    fetch page summaries. By default, it will return the page summaries
    of the top-k results.
    It limits the Document content by doc_content_chars_max.
    """

    wiki_client: Any  #: :meta private:

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that the python package exists in environment."""
        try:
            import wikipedia

            lang = values.get("lang", "en")
            wikipedia.set_lang(lang)
            values["wiki_client"] = wikipedia
        except ImportError:
            raise ImportError(
                "Could not import wikipedia python package. "
                "Please install it with `pip install wikipedia`."
            )
        return values

    def search(self, query: str, top_k_results: int = 3, doc_content_chars_max: int = 4000) -> str:
        """Search Wikipedia and get page summaries."""
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=top_k_results
        )
        summaries = []
        for page_title in page_titles[:top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[:doc_content_chars_max]
    
    def fetch(self, page_title: str, doc_content_chars_max: int = 20000) -> str:
        """Fetch a specific Wikipedia page by title and return the full article. Returns the closest match if the page is ambiguous."""
        page_titles = self.wiki_client.search(
            page_title[:WIKIPEDIA_MAX_QUERY_LENGTH], results=1
        )
        if wiki_page := self._fetch_page(page_titles[0]):
            article_text = f"Page: {page_titles[0]}\n\n{wiki_page.content[:doc_content_chars_max]}"
            return article_text
        return f"No Wikipedia page found for '{page_titles[0]}'. Try using the search tool."


    @staticmethod
    def _formatted_page_summary(page_title: str, wiki_page: Any) -> Optional[str]:
        return f"Page: {page_title}\nSummary: {wiki_page.summary}"

    def _page_to_document(self, page_title: str, wiki_page: Any, 
                          load_all_available_meta: bool = False, 
                          doc_content_chars_max: int = 4000) -> Document:
        main_meta = {
            "title": page_title,
            "summary": wiki_page.summary,
            "source": wiki_page.url,
        }
        add_meta = (
            {
                "categories": wiki_page.categories,
                "page_url": wiki_page.url,
                "image_urls": wiki_page.images,
                "related_titles": wiki_page.links,
                "parent_id": wiki_page.parent_id,
                "references": wiki_page.references,
                "revision_id": wiki_page.revision_id,
                "sections": wiki_page.sections,
            }
            if load_all_available_meta
            else {}
        )
        doc = Document(
            page_content=wiki_page.content[:doc_content_chars_max],
            metadata={
                **main_meta,
                **add_meta,
            },
        )
        return doc

    def _fetch_page(self, page: str) -> Optional[str]:
        try:
            return self.wiki_client.page(title=page, auto_suggest=False)
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            return None

    def load(self, query: str, top_k_results: int = 3, 
             load_all_available_meta: bool = False, 
             doc_content_chars_max: int = 4000) -> List[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.

        Returns: a list of documents.
        """
        return list(self.lazy_load(
            query, 
            top_k_results=top_k_results, 
            load_all_available_meta=load_all_available_meta, 
            doc_content_chars_max=doc_content_chars_max
        ))

    def lazy_load(self, query: str, top_k_results: int = 3, 
                 load_all_available_meta: bool = False, 
                 doc_content_chars_max: int = 4000) -> Iterator[Document]:
        """
        Run Wikipedia search and get the article text plus the meta information.

        Returns: a list of documents.
        """
        page_titles = self.wiki_client.search(
            query[:WIKIPEDIA_MAX_QUERY_LENGTH], results=top_k_results
        )
        for page_title in page_titles[:top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if doc := self._page_to_document(
                    page_title, 
                    wiki_page, 
                    load_all_available_meta=load_all_available_meta,
                    doc_content_chars_max=doc_content_chars_max
                ):
                    yield doc


class WikipediaSearchInput(BaseModel):
    """Input for the Wikipedia search tool."""

    query: str = Field(..., description="The search query to search Wikipedia for.")
    top_k_results: int = Field(3, description="The number of search results to return.")
    doc_content_chars_max: int = Field(4000, description="The maximum number of characters to return for each document.")


class WikipediaFetchInput(BaseModel):
    """Input for the Wikipedia fetch tool."""

    page_title: str = Field(..., description="The title of the Wikipedia page to fetch.")
    doc_content_chars_max: int = Field(20000, description="The maximum number of characters to return for the article.")


class WikipediaSearchTool(BaseTool):
    """Tool that searches Wikipedia and returns summaries."""

    name: str = "wikipedia_search"
    description: str = (
        "A wrapper around Wikipedia search. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query. Returns summaries of the top search results."
    )
    api_wrapper: WikipediaAPIWrapper

    args_schema: Type[BaseModel] = WikipediaSearchInput

    def _run(
        self,
        query: str,
        top_k_results: int = 3,
        doc_content_chars_max: int = 4000,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia search tool."""
        return self.api_wrapper.search(
            query,
            top_k_results=top_k_results,
            doc_content_chars_max=doc_content_chars_max
        )


class WikipediaFetchTool(BaseTool):
    """Tool that fetches a specific Wikipedia page by title."""

    name: str = "wikipedia_fetch"
    description: str = (
        "Retrieve a specific Wikipedia page by its title. "
        "Useful for when you need comprehensive information about "
        "people, places, companies, facts, historical events, or other subjects. "
        "This returns the complete article text rather than just summary. "
        "Returns the closest match if the page is ambiguous."
    )
    api_wrapper: WikipediaAPIWrapper

    args_schema: Type[BaseModel] = WikipediaFetchInput

    def _run(
        self,
        page_title: str,
        doc_content_chars_max: int = 20000,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Wikipedia fetch tool."""
        return self.api_wrapper.fetch(
            page_title,
            doc_content_chars_max=doc_content_chars_max
        )


# Wikipedia Toolkit
class WikipediaToolkit:
    """Toolkit for Wikipedia."""
    
    def __init__(self):
        self.api_wrapper = WikipediaAPIWrapper()
    
    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            WikipediaSearchTool(api_wrapper=self.api_wrapper),
            WikipediaFetchTool(api_wrapper=self.api_wrapper),
        ]
    

# wikipedia = WikipediaAPIWrapper(doc_content_chars_max = 400000)
# print(wikipedia.fetch("India Pakistan conflict 2025"))