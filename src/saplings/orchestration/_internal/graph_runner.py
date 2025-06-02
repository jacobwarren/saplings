from __future__ import annotations

"""
GraphRunner module for Saplings.

This module provides the GraphRunner class for coordinating multiple agents.
"""


import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from saplings.api.models import LLM

from saplings.orchestration._internal.config import (
    AgentNode,
    CommunicationChannel,
    GraphRunnerConfig,
    NegotiationStrategy,
)

logger = logging.getLogger(__name__)


class GraphRunner:
    """
    Coordinator for multiple agents in a graph structure.

    This class provides functionality for:
    - Registering agents and defining relationships
    - Implementing debate and contract-net negotiation strategies
    - Executing multi-agent workflows
    - Tracking agent interactions
    """

    def __init__(
        self,
        model: "LLM",
        config: GraphRunnerConfig | None = None,
    ) -> None:
        """
        Initialize the graph runner.

        Args:
        ----
            model: LLM model to use for coordination
            config: Configuration for the graph runner

        """
        self.model = model
        self.config = config or GraphRunnerConfig()

        # Initialize agent graph
        self.agents: dict[str, AgentNode] = {}
        self.channels: list[CommunicationChannel] = []

        # Initialize interaction history
        self.history: list[dict[str, Any]] = []

        # Initialize memory if provided or enabled
        from saplings.api.memory import MemoryStore

        self.memory_store = self.config.memory_store or MemoryStore()

        # Initialize monitoring if enabled
        if self.config.enable_monitoring:
            from saplings.api.monitoring import BlameGraph, TraceManager

            # Use the provided trace manager or create a new one
            if self.config.trace_manager:
                self.trace_manager = self.config.trace_manager
            else:
                self.trace_manager = TraceManager()

            # Use the provided blame graph or create a new one
            if self.config.blame_graph:
                self.blame_graph = self.config.blame_graph
            else:
                self.blame_graph = BlameGraph(trace_manager=self.trace_manager)
        else:
            self.trace_manager = None
            self.blame_graph = None

        # Initialize validation if enabled
        if self.config.enable_validation:
            from saplings.api.judge import JudgeAgent

            # Use the provided judge or create a new one
            if self.config.judge:
                self.judge = self.config.judge
            else:
                # Create a new judge with the model
                self.judge = JudgeAgent(model=model)
        else:
            self.judge = None

        # Initialize self-healing if enabled
        if self.config.enable_self_healing:
            from saplings.api.self_heal import SuccessPairCollector

            self.success_pair_collector = (
                self.config.success_pair_collector or SuccessPairCollector()
            )
        else:
            self.success_pair_collector = None

    def register_agent(self, agent: AgentNode) -> None:
        """
        Register an agent in the graph.

        Args:
        ----
            agent: Agent to register

        Raises:
        ------
            ValueError: If an agent with the same ID already exists

        """
        if agent.id in self.agents:
            msg = f"Agent with ID '{agent.id}' already exists"
            raise ValueError(msg)

        self.agents[agent.id] = agent

        # Log registration
        logger.info(f"Registered agent: {agent.name} (ID: {agent.id})")

    def create_channel(
        self,
        source_id: str,
        target_id: str,
        channel_type: str = "direct",
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CommunicationChannel:
        """
        Create a communication channel between agents.

        Args:
        ----
            source_id: ID of the source agent
            target_id: ID of the target agent
            channel_type: Type of channel
            description: Description of the channel
            metadata: Additional metadata

        Returns:
        -------
            CommunicationChannel: The created channel

        Raises:
        ------
            ValueError: If either agent does not exist

        """
        # Verify that both agents exist
        if source_id not in self.agents:
            msg = f"Agent with ID '{source_id}' does not exist"
            raise ValueError(msg)
        if target_id not in self.agents:
            msg = f"Agent with ID '{target_id}' does not exist"
            raise ValueError(msg)

        # Create the channel
        channel = CommunicationChannel(
            source=source_id,
            target=target_id,
            channel_type=channel_type,
            metadata=metadata or {},
        )

        # Add the channel to the list
        self.channels.append(channel)

        logger.info(
            f"Created channel: {channel_type} from {source_id} to {target_id} ({description})"
        )

        return channel

    # Memory and retrieval methods
    def add_to_memory(self, document: Any) -> None:
        """
        Add a document to the shared memory store.

        Args:
        ----
            document: Document to add

        """
        if self.memory_store:
            self.memory_store.add_document(document)
            logger.info(f"Added document to shared memory: {getattr(document, 'id', 'unknown')}")
        else:
            logger.warning("No memory store available")

    def retrieve_from_memory(self, query: str, limit: int = 5) -> list[Any]:
        """
        Retrieve documents from the shared memory store.

        Args:
        ----
            query: Query to search for
            limit: Maximum number of documents to retrieve

        Returns:
        -------
            List[Any]: Retrieved documents

        """
        if self.memory_store:
            # Create a simple text document for the query
            from saplings.api.memory.document import Document, DocumentMetadata

            # Create metadata for the query document
            metadata = DocumentMetadata(
                source="query", content_type="text/plain", language="en", author="system"
            )

            # Create the document
            query_doc = Document(id="query", content=query, metadata=metadata)

            # Use the indexer to create an embedding
            if not query_doc.embedding:
                self.memory_store.indexer.index_document(query_doc)

            # Search using the embedding
            if query_doc.embedding is not None:
                results = self.memory_store.search(query_embedding=query_doc.embedding, limit=limit)
                logger.info(f"Retrieved {len(results)} documents from shared memory")
                return [doc for doc, _ in results]  # Unpack (document, score) tuples
            logger.warning("Failed to create embedding for query")
            return []
        logger.warning("No memory store available")
        return []

    # Monitoring and tracing methods
    def create_trace(self, task: str) -> str | None:
        """
        Create a new trace for a task.

        Args:
        ----
            task: Task being traced

        Returns:
        -------
            Optional[str]: Trace ID if monitoring is enabled, None otherwise

        """
        if self.trace_manager:
            trace_id = str(uuid.uuid4())
            self.trace_manager.create_trace(trace_id=trace_id, attributes={"task": task})
            logger.info(f"Created trace {trace_id} for task: {task}")
            return trace_id
        return None

    def add_span(
        self, trace_id: str | None, agent_id: str, action: str, content: str
    ) -> str | None:
        """
        Add a span to a trace.

        Args:
        ----
            trace_id: ID of the trace
            agent_id: ID of the agent
            action: Action being performed
            content: Content of the action

        Returns:
        -------
            Optional[str]: Span ID if monitoring is enabled, None otherwise

        """
        if self.trace_manager and trace_id:
            span_id = str(uuid.uuid4())
            span = self.trace_manager.start_span(
                name=f"{agent_id}_{action}",
                trace_id=trace_id,
                attributes={
                    "agent_id": agent_id,
                    "action": action,
                    "content_length": len(content),
                },
            )
            # Store the span ID for later use
            span_id = span.span_id if hasattr(span, "span_id") else span_id
            logger.debug(f"Added span {span_id} to trace {trace_id}")
            return span_id
        return None

    # Validation and judging methods
    async def validate_output(self, output: str | None, prompt: str) -> dict[str, Any] | None:
        """
        Validate an output using the judge agent.

        Args:
        ----
            output: Output to validate
            prompt: Prompt that generated the output

        Returns:
        -------
            Optional[Dict[str, Any]]: Validation result if validation is enabled, None otherwise

        """
        if self.judge and output:
            result = await self.judge.judge(output, prompt)
            logger.info(f"Validated output: passed={result.passed}, score={result.overall_score}")
            return {
                "passed": result.passed,
                "score": result.overall_score,
                "critique": result.critique,
                "suggestions": result.suggestions,
            }
        return None

    # Self-healing methods
    async def collect_success_pair(
        self, input_text: str, output_text: str | None, score: float
    ) -> None:
        """
        Collect a successful input-output pair.

        Args:
        ----
            input_text: Input text
            output_text: Output text
            score: Quality score

        """
        # Threshold for high-quality pairs
        HIGH_QUALITY_THRESHOLD = 0.8

        if (
            self.success_pair_collector and output_text and score >= HIGH_QUALITY_THRESHOLD
        ):  # Only collect high-quality pairs
            await self.success_pair_collector.collect(
                input_text=input_text,
                output_text=output_text,
                metadata={
                    "score": score,
                    "timestamp": time.time(),
                },
            )
            logger.info(f"Collected success pair with score {score}")

    async def negotiate(
        self,
        task: str,
        context: str | None = None,
        max_rounds: int | None = None,
        timeout_seconds: int | None = None,
    ) -> str | None:
        """
        Run a negotiation between agents to solve a task.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of negotiation rounds (overrides config)
            timeout_seconds: Timeout for negotiation in seconds (overrides config)

        Returns:
        -------
            str: Result of the negotiation

        Raises:
        ------
            ValueError: If the negotiation strategy is invalid
            asyncio.TimeoutError: If the negotiation times out

        """
        # Use provided values or fall back to config
        max_rounds = max_rounds or self.config.max_iterations
        timeout_seconds = timeout_seconds or self.config.timeout_seconds

        # Clear the history
        self.history = []

        # Create a trace if monitoring is enabled
        trace_id = self.create_trace(task)

        # Log the start of negotiation
        logger.info(
            f"Starting negotiation for task: {task} "
            f"(Strategy: {self.config.negotiation_strategy}, "
            f"Max rounds: {max_rounds}, Timeout: {timeout_seconds}s)"
        )

        # Run the appropriate negotiation strategy with a timeout
        try:
            result = await asyncio.wait_for(
                self._run_negotiation(task, context, max_rounds, trace_id),
                timeout=timeout_seconds,
            )

            # Log the result
            logger.info(f"Negotiation completed: {result}")

            # Validate the result if validation is enabled
            if self.config.enable_validation and self.judge:
                validation = await self.validate_output(result, task)
                if validation:
                    # Collect success pair if self-healing is enabled
                    if self.config.enable_self_healing and validation["passed"]:
                        await self.collect_success_pair(task, result, validation["score"])

                    # Add validation information to the result
                    result = f"Result: {result}\n\nValidation: {json.dumps(validation, indent=2)}"

            return result
        except asyncio.TimeoutError:
            logger.warning(f"Negotiation timed out after {timeout_seconds}s")
            raise

    async def _run_negotiation(
        self,
        task: str,
        context: str | None,
        max_rounds: int,  # Used to determine the maximum number of negotiation rounds
        trace_id: str | None = None,
    ) -> str | None:
        """
        Run the appropriate negotiation strategy.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of negotiation rounds
            trace_id: ID of the trace for monitoring

        Returns:
        -------
            str: Result of the negotiation

        Raises:
        ------
            ValueError: If the negotiation strategy is invalid

        """
        if self.config.negotiation_strategy == NegotiationStrategy.DEBATE:
            return await self._run_debate(task, context, max_rounds, trace_id)
        if self.config.negotiation_strategy == NegotiationStrategy.CONTRACT_NET:
            return await self._run_contract_net(task, context, max_rounds, trace_id)
        msg = f"Invalid negotiation strategy: {self.config.negotiation_strategy}"
        raise ValueError(msg)

    async def _run_debate(
        self,
        task: str,
        context: str | None,
        max_rounds: int,  # Used to limit the number of debate rounds
        trace_id: str | None = None,
    ) -> str | None:
        """
        Run a debate between agents to reach consensus.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of debate rounds
            trace_id: ID of the trace for monitoring

        Returns:
        -------
            str: Result of the debate

        """
        # Verify that we have at least two agents
        MIN_AGENTS_FOR_DEBATE = 2
        if len(self.agents) < MIN_AGENTS_FOR_DEBATE:
            msg = "Debate requires at least two agents"
            raise ValueError(msg)

        # Initialize the debate
        round_num = 0
        consensus_reached = False
        current_proposal = ""

        # Run the debate for up to max_rounds
        while round_num < max_rounds and not consensus_reached:
            round_num += 1
            logger.info(f"Starting debate round {round_num}/{max_rounds}")

            # Get all agents involved in the debate
            debate_agents = list(self.agents.values())

            # In the first round, the first agent makes a proposal
            if round_num == 1:
                proposer = debate_agents[0]
                proposal = await self._generate_proposal(proposer, task, context, trace_id)
                current_proposal = proposal
                self.history.append(
                    {
                        "round": round_num,
                        "agent": proposer.id,
                        "action": "propose",
                        "content": proposal,
                    }
                )
                logger.info(
                    f"Agent {proposer.id} proposed: {proposal[:100] if proposal else ''}..."
                )
            else:
                # In subsequent rounds, agents critique and refine the proposal
                critiques = []
                for agent in debate_agents:
                    critique = await self._generate_critique(
                        agent, task, current_proposal, self.history, trace_id
                    )
                    critiques.append(critique)
                    self.history.append(
                        {
                            "round": round_num,
                            "agent": agent.id,
                            "action": "critique",
                            "content": critique,
                        }
                    )
                    logger.info(
                        f"Agent {agent.id} critiqued: {critique[:100] if critique else ''}..."
                    )

                # Generate a refined proposal based on critiques
                refiner = debate_agents[round_num % len(debate_agents)]
                refined_proposal = await self._generate_refinement(
                    refiner, task, current_proposal, critiques, trace_id
                )
                current_proposal = refined_proposal
                self.history.append(
                    {
                        "round": round_num,
                        "agent": refiner.id,
                        "action": "refine",
                        "content": refined_proposal,
                    }
                )
                logger.info(
                    f"Agent {refiner.id} refined: {refined_proposal[:100] if refined_proposal else ''}..."
                )

                # Check if consensus has been reached
                consensus = await self._check_consensus(
                    debate_agents, task, current_proposal, trace_id
                )
                if consensus:
                    consensus_reached = True
                    logger.info("Consensus reached")

        # Return the final proposal
        return current_proposal

    async def _run_contract_net(
        self,
        task: str,
        context: str | None,
        max_rounds: int,  # Not used in this implementation but kept for API consistency
        trace_id: str | None = None,
    ) -> str | None:
        """
        Run a contract net protocol to delegate tasks.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of rounds
            trace_id: ID of the trace for monitoring

        Returns:
        -------
            str: Result of the contract net protocol

        """
        # Simplified implementation of contract net protocol
        logger.info("Running contract net protocol")

        # For now, just use the first agent as the manager
        manager = list(self.agents.values())[0]

        # Manager announces the task
        announcement = f"Task: {task}\nContext: {context or 'None'}"
        self.history.append(
            {
                "round": 1,
                "agent": manager.id,
                "action": "announce",
                "content": announcement,
            }
        )
        logger.info(f"Manager {manager.id} announced task")

        # Collect bids from other agents
        bids = []
        for agent_id, agent in self.agents.items():
            if agent_id != manager.id:
                bid = await self._generate_bid(agent, task, context, trace_id)
                bids.append({"agent": agent, "bid": bid})
                self.history.append(
                    {
                        "round": 1,
                        "agent": agent_id,
                        "action": "bid",
                        "content": bid,
                    }
                )
                logger.info(f"Agent {agent_id} submitted bid: {bid[:100] if bid else ''}...")

        # Manager selects the best bid
        if bids:
            selected_bid = await self._select_bid(manager, task, bids, trace_id)
            selected_agent = selected_bid["agent"]
            self.history.append(
                {
                    "round": 1,
                    "agent": manager.id,
                    "action": "select",
                    "content": f"Selected agent: {selected_agent.id}",
                }
            )
            logger.info(f"Manager {manager.id} selected agent {selected_agent.id}")

            # Selected agent performs the task
            result = await self._perform_task(selected_agent, task, context, trace_id)
            self.history.append(
                {
                    "round": 2,
                    "agent": selected_agent.id,
                    "action": "perform",
                    "content": result,
                }
            )
            logger.info(
                f"Agent {selected_agent.id} performed task: {result[:100] if result else ''}..."
            )

            return result

        # If no bids, manager performs the task
        result = await self._perform_task(manager, task, context, trace_id)
        self.history.append(
            {
                "round": 2,
                "agent": manager.id,
                "action": "perform",
                "content": result,
            }
        )
        logger.info(f"Manager {manager.id} performed task: {result[:100] if result else ''}...")

        return result

    # Helper methods for debate
    async def _generate_proposal(
        self, agent: AgentNode, task: str, context: str | None, trace_id: str | None = None
    ) -> str | None:
        """Generate a proposal for a task."""
        prompt = f"""
        Task: {task}

        Context: {context or 'None'}

        You are {agent.name}. Please generate a detailed proposal to solve this task.
        Your proposal should be comprehensive and address all aspects of the task.
        """

        # Add span if tracing is enabled
        span_id = (
            self.add_span(trace_id, agent.id, "generate_proposal", prompt) if trace_id else None
        )

        # Generate the proposal
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        return response.content

    async def _generate_critique(
        self,
        agent: AgentNode,
        task: str,
        proposal: str | None,
        history: list[dict[str, Any]],
        trace_id: str | None = None,
    ) -> str | None:
        """Generate a critique of a proposal."""
        # Format the history for the prompt
        history_text = "\n".join(
            [
                f"Round {entry['round']}: {entry['agent']} {entry['action']}: {entry['content'][:100]}..."
                for entry in history
            ]
        )

        prompt = f"""
        Task: {task}

        Current Proposal:
        {proposal}

        History:
        {history_text}

        You are {agent.name}. Please critique the current proposal.
        Identify strengths, weaknesses, and areas for improvement.
        Be constructive and specific in your feedback.
        """

        # Add span if tracing is enabled
        span_id = (
            self.add_span(trace_id, agent.id, "generate_critique", prompt) if trace_id else None
        )

        # Generate the critique
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        return response.content

    async def _generate_refinement(
        self,
        agent: AgentNode,
        task: str,
        proposal: str | None,
        critiques: list[str],
        trace_id: str | None = None,
    ) -> str | None:
        """Generate a refined proposal based on critiques."""
        # Format the critiques for the prompt
        critiques_text = "\n\n".join(
            [f"Critique {i+1}:\n{critique}" for i, critique in enumerate(critiques)]
        )

        prompt = f"""
        Task: {task}

        Current Proposal:
        {proposal}

        Critiques:
        {critiques_text}

        You are {agent.name}. Please refine the current proposal based on the critiques.
        Address the weaknesses identified and incorporate the suggestions.
        Your refined proposal should be comprehensive and address all aspects of the task.
        """

        # Add span if tracing is enabled
        span_id = (
            self.add_span(trace_id, agent.id, "generate_refinement", prompt) if trace_id else None
        )

        # Generate the refinement
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        return response.content

    async def _check_consensus(
        self, agents: list[AgentNode], task: str, proposal: str | None, trace_id: str | None = None
    ) -> bool:
        """Check if consensus has been reached on a proposal."""
        consensus_count = 0
        threshold = len(agents) * 0.75  # 75% of agents must agree

        for agent in agents:
            prompt = f"""
            Task: {task}

            Current Proposal:
            {proposal}

            You are {agent.name}. Do you agree with the current proposal?
            Please answer with 'Yes' or 'No' and provide a brief explanation.
            """

            # Add span if tracing is enabled
            span_id = (
                self.add_span(trace_id, agent.id, "check_consensus", prompt) if trace_id else None
            )

            # Generate the response
            response = await self.model.generate(prompt)

            # End span if tracing is enabled
            if span_id and self.trace_manager:
                self.trace_manager.end_span(span_id)

            # Check if the agent agrees
            if response and response.content and "yes" in response.content.lower():
                consensus_count += 1

        return consensus_count >= threshold

    # Helper methods for contract net
    async def _generate_bid(
        self, agent: AgentNode, task: str, context: str | None, trace_id: str | None = None
    ) -> str | None:
        """Generate a bid for a task."""
        prompt = f"""
        Task: {task}

        Context: {context or 'None'}

        You are {agent.name}. Please generate a bid for this task.
        Your bid should include:
        1. Your qualifications for the task
        2. Your approach to solving the task
        3. Your estimated time to complete the task
        4. Any resources you would need
        """

        # Add span if tracing is enabled
        span_id = self.add_span(trace_id, agent.id, "generate_bid", prompt) if trace_id else None

        # Generate the bid
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        return response.content

    async def _select_bid(
        self, manager: AgentNode, task: str, bids: list[dict[str, Any]], trace_id: str | None = None
    ) -> dict[str, Any]:
        """Select the best bid for a task."""
        # Format the bids for the prompt
        bids_text = "\n\n".join(
            [f"Bid from {bid['agent'].name} (ID: {bid['agent'].id}):\n{bid['bid']}" for bid in bids]
        )

        prompt = f"""
        Task: {task}

        Bids:
        {bids_text}

        You are {manager.name}, the manager. Please select the best bid for this task.
        Analyze each bid based on:
        1. Qualifications for the task
        2. Approach to solving the task
        3. Estimated time to complete
        4. Required resources

        Respond with the ID of the selected agent and a brief explanation.
        """

        # Add span if tracing is enabled
        span_id = self.add_span(trace_id, manager.id, "select_bid", prompt) if trace_id else None

        # Generate the selection
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        # Parse the response to find the selected agent ID
        for bid in bids:
            if bid["agent"].id in response.content:
                return bid

        # If no agent ID is found, return the first bid
        return bids[0]

    async def _perform_task(
        self, agent: AgentNode, task: str, context: str | None, trace_id: str | None = None
    ) -> str | None:
        """Perform a task."""
        prompt = f"""
        Task: {task}

        Context: {context or 'None'}

        You are {agent.name}. Please perform this task and provide a detailed solution.
        Your solution should be comprehensive and address all aspects of the task.
        """

        # Add span if tracing is enabled
        span_id = self.add_span(trace_id, agent.id, "perform_task", prompt) if trace_id else None

        # Generate the solution
        response = await self.model.generate(prompt)

        # End span if tracing is enabled
        if span_id and self.trace_manager:
            self.trace_manager.end_span(span_id)

        return response.content
