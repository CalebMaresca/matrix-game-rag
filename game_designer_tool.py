from langchain_core.tools import BaseTool, ArgsSchema
from langchain_core.callbacks import CallbackManagerForToolRun
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal, List, Dict, Optional, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, BasePromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableSequence

class ForceUnit(BaseModel):
    unit_name: str = Field(description="The name or type of the unit/asset (e.g., 'Motorised Infantry Regiment 3', 'Task Force 79.1', 'Submarine Force', 'Chemical Weapons', 'Public Support Unit').")
    location: str = Field(description="The starting location of the unit on the game map (e.g., 'Stanley', 'Argentina', 'Eastern Mediterranean', 'Sarajevo', 'Kaliningrad').")
    details: Optional[str] = Field(None, description="Any specific notes, stats, or rules associated with this unit (e.g., 'Elite', '-1 on combat dice rolls', 'Hidden at start', 'Requires argument to move').")

class Actor(BaseModel):
    actor_name: str = Field(description="The specific name of the player/faction/role (e.g., 'UK Government - Mrs. Thatcher', 'Russia', 'Tribal Elder').")
    actor_briefing: str = Field(description="A detailed description of the actor's perspective, motivations, capabilities, constraints, relationships with other actors, and current situation within the game's context. Provide enough information for a player (or AI) to effectively role-play this actor. This briefing will only be shown to the player, not the other actors and can contain secret information.")
    objectives: List[str] = Field(description="A list of 3-5 specific, measurable, achievable, relevant, and time-bound (SMART-like, though flexibility is key in matrix games) objectives for this actor to pursue during the game. Objectives should create potential conflict and interaction with other actors.")
    starting_forces: List[ForceUnit] = Field(description="A list of units, assets, resources, or capabilities controlled by this actor at the start of the game, including their starting locations if applicable.")


class MatrixGame(BaseModel):
    name: str = Field(description="Title of the game scenario")
    introduction: str = Field(description="A brief, engaging paragraph setting the stage for the game. Introduce the core conflict or situation the game simulates.")
    background_briefing: str = Field(description="Detailed context and background information necessary to understand the scenario. Include historical context, key events leading up to the game's start, the current geopolitical situation, and any relevant factors influencing the actors and the conflict. Should provide enough detail for players to understand the world they are stepping into.")
    actors: List[Actor] = Field(description="An ordered list defining the key players or factions involved in the game. Each actor represents a distinct entity (nation, organization, group, individual) whose actions drive the narrative. The list should be ordered by the sequence of play.")
    victory_conditions: Optional[str] = Field(default=None, description="How the game ends and how winners are determined, if applicable")
    turn_length: str = Field(description="The length of a turn in the game")
    game_length: int = Field(description="The maximum number of turns the game can last")
    designer_notes: Optional[str] = Field(default=None, description="Optional insights about the game's design purpose, expected outcomes, or historical parallels")

class GameDesignerInput(BaseModel):
    """Input for the game designer tool."""

    prompt: str = Field(description="The prompt for the game designer. This should be a detailed description of the game you want to design.")

SYSTEM_PROMPT = """
You are an expert Matrix Game Designer. Your task is to take a user's description of a desired game scenario and transform it into a complete, well-structured `MatrixGame` object, adhering strictly to the defined format and embodying the core principles of matrix gaming.

**Core Matrix Game Principles to Embody:**

* **Focus on Narrative & Insight:** The primary goal is not necessarily to have clear winners and losers, but to generate a credible narrative through player actions and arguments, fostering analytical understanding and insight into the simulated situation.
* **Plausible Actions & Structured Arguments:** The game revolves around players proposing plausible actions ("Arguments") supported by reasons ("Pros"). Success is often determined by structured argument and discussion, sometimes involving adjudication (like dice or voting), rather than complex, pre-defined rules.
* **Minimal Rules:** Avoid overly complex mechanics. The strength lies in the open-ended nature driven by player arguments...
* **Role-Playing:** Actors should be designed to encourage players to step into their roles, considering their unique perspectives, motivations, and constraints.
* **Realism:** Unless otherwise specified, the game should be designed to be as realistic as possible, with the goal of providing strategic insights to the user.

**Instructions for Generating the MatrixGame Object:**

Based on the user's input prompt, you must generate the following fields precisely according to the specified descriptions:

1.  **`name` (str):** Create a concise and engaging title for the game scenario.
2.  **`introduction` (str):** Write a brief, engaging paragraph (2-4 sentences) that sets the stage and introduces the core conflict or situation being simulated.
3.  **`background_briefing` (str):** Provide detailed background information essential for understanding the scenario. Include historical context, key events leading up, the current geopolitical/situational landscape, and relevant factors influencing the actors. Ensure enough detail for players to understand the world state at the start of the game. Keep it relatively concise but comprehensive.
4.  **`actors` (List[Actor]):** This is a crucial part. Define a list of the key Actors (players/factions).
    * **Selection & Balance:** Choose 6-8 actors if possible, ensuring a balance of opposing views and capabilities to foster interesting interactions and avoid one-sided scenarios. Consider main protagonists, key supporters (perhaps with divergent goals), and relevant third parties or opposition groups. The list order determines the initial turn sequence.
    * For each `Actor`:
        * **`actor_name` (str):** A specific, descriptive name (e.g., 'UK Government - Mrs. Thatcher', 'Russia', 'Tribal Elder', 'Anonymous Hacking Collective').
        * **`actor_briefing` (str):** This is the most critical part for role-playing. Write a detailed but concise (ideally ~1 page equivalent) briefing *from the actor's perspective*. It must cover their motivations, worldview, capabilities (political, military, economic, social etc.), constraints, relationships with others, and current situation. Include any *secret information* or hidden starting conditions/knowledge relevant only to this actor. The briefing should be interesting and provide enough information for effective role-play.
        * **`objectives` (List[str]):** Define 3-5 specific, SMART-like objectives for the actor. These objectives *must* be designed to create potential conflict, tension, and interaction with other actors' objectives. Phrase them actively (e.g., "Seize control of the oil fields!", "Prevent secession!"). Include at least one longer-term objective if appropriate for the scenario.
        * **`starting_forces` (List[ForceUnit]):** List the tangible units, assets, resources, or key capabilities controlled by the actor at the game's start.
            * For each `ForceUnit`:
                * **`unit_name` (str):** Clear name (e.g., 'Motorised Infantry Regiment 3', 'Task Force 79.1', 'Public Support Unit', 'Key Media Outlet').
                * **`location` (str):** Starting location on the map, if applicable.
                * **`details` (Optional[str]):** Any special notes, stats, or rules (e.g., 'Elite', '-1 on combat dice rolls', 'Hidden at start', 'Requires argument to move').
5.  **`victory_conditions` (Optional[str]):** Define how the game might end or how success is viewed. Note that matrix games often don't have traditional winners/losers; success might be about achieving objectives or gaining insights. If there are specific conditions, state them clearly. Otherwise, state that success is based on achieving objectives or note that it's determined post-game based on the narrative.
6.  **`turn_length` (str):** Specify the approximate real-world time each game turn represents (e.g., "One Week", "2-4 Weeks", "One Month"). This should be appropriate for the scenario's scope and actions.
7.  **`game_length` (int):** Specify the maximum number of turns. Aim for at least 6 turns to allow for action-reaction cycles[cite: 119, 233].
8.  **`designer_notes` (Optional[str]):** Optionally, add brief notes about the design intent, potential learning outcomes, historical parallels, or key dynamics to watch for.

**Output Format:**

Your final output MUST be the generated `MatrixGame` object, strictly conforming to the structure and types defined. Do not include any conversational text before or after the structured output. Ensure all descriptions within the object fields are clear, well-written, and fulfill the requirements outlined above.
"""

class GameDesignerTool(BaseTool):
    """Tool that designs matrix games."""

    name: str = "game_designer"
    description: str = (
        "Useful for when you need to design a matrix game."
        "Input should be a detailed description of the game you want to design."
        "The output will be a structured description of the game, including the game's name, introduction, background briefing, actors, victory conditions, turn length, and game length."
        "After using this tool, always present the output to the user in a structured markdown format."
    )
    args_schema: Optional[ArgsSchema] = GameDesignerInput
    # return_direct: bool = True
    
    model: BaseChatModel = Field(default=None, exclude=True)
    system_prompt: str = Field(default=SYSTEM_PROMPT, exclude=True)
    prompt: ChatPromptTemplate = Field(default=None, exclude=True)
    chain: RunnableSequence = Field(default=None, exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = ChatOpenAI(model="gpt-4.1").with_structured_output(MatrixGame)
        self.system_prompt = SYSTEM_PROMPT
        self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("user", "{prompt}"),
            ])
        self.chain = self.prompt | self.model

    def _run(
        self,
        prompt: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the game designer tool."""

        response = self.chain.invoke(
                {"prompt": prompt}
            )
        return response