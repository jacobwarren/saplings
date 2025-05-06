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
from typing import Any

from saplings.core.model_adapter import LLM
from saplings.judge import JudgeAgent
from saplings.memory import MemoryStore
from saplings.monitoring import BlameGraph, TraceManager
from saplings.orchestration.config import (
    AgentNode,
    CommunicationChannel,
    GraphRunnerConfig,
    NegotiationStrategy,
)
from saplings.self_heal import SuccessPairCollector

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
        model: LLM,
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
        self.memory_store = self.config.memory_store or MemoryStore()

        # Initialize monitoring if enabled
        if self.config.enable_monitoring:
            self.trace_manager = self.config.trace_manager or TraceManager()
            self.blame_graph = self.config.blame_graph or BlameGraph(
                trace_manager=self.trace_manager
            )
        else:
            self.trace_manager = None
            self.blame_graph = None

        # Initialize validation if enabled
        if self.config.enable_validation:
            self.judge = self.config.judge or JudgeAgent(model=model)
        else:
            self.judge = None

        # Initialize self-healing if enabled
        if self.config.enable_self_healing:
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

        # If the agent has a memory store, connect it to the shared memory
        if agent.memory_store is None and self.memory_store:
            agent.memory_store = self.memory_store

        # If GASA is enabled for this agent but no config is provided, use default
        if agent.enable_gasa and agent.gasa_config is None:
            agent.gasa_config = {"max_hops": 2, "mask_strategy": "binary"}

        # Log additional information based on agent configuration
        if agent.agent:
            logger.info(
                f"Registered agent: {agent.name} (ID: {agent.id}, Role: {agent.role}, Using base Agent)"
            )
        elif agent.provider and agent.model:
            logger.info(
                f"Registered agent: {agent.name} (ID: {agent.id}, Role: {agent.role}, Model: {agent.provider}/{agent.model})"
            )
        else:
            logger.info(f"Registered agent: {agent.name} (ID: {agent.id}, Role: {agent.role})")

    def create_channel(
        self,
        source_id: str,
        target_id: str,
        channel_type: str,
        description: str,
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
            source_id=source_id,
            target_id=target_id,
            channel_type=channel_type,
            description=description,
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
            from saplings.memory.document import Document, DocumentMetadata

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

    def add_span(self, trace_id: str, agent_id: str, action: str, content: str) -> str | None:
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
            self.trace_manager.start_span(
                name=f"{agent_id}_{action}",
                trace_id=trace_id,
                attributes={
                    "agent_id": agent_id,
                    "action": action,
                    "content_length": len(content),
                },
            )
            logger.debug(f"Added span {span_id} to trace {trace_id}")
            return span_id
        return None

    # Validation and judging methods
    async def validate_output(self, output: str, prompt: str) -> dict[str, Any] | None:
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
        if self.judge:
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
    async def collect_success_pair(self, input_text: str, output_text: str, score: float) -> None:
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
            self.success_pair_collector and score >= HIGH_QUALITY_THRESHOLD
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
    ) -> str:
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
        max_rounds = max_rounds or self.config.max_rounds
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
        max_rounds: int,
        trace_id: str | None = None,
    ) -> str:
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
        max_rounds: int,
        trace_id: str | None = None,
    ) -> str:
        """
        Run a debate between agents to reach consensus.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of debate rounds

        Returns:
        -------
            str: Result of the debate

        """
        # Verify that we have at least two agents
        # Minimum number of agents required for debate
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
            # Index for round num
            ROUND_NUM_INDEX = 1

            if round_num == ROUND_NUM_INDEX:
                proposer = debate_agents[0]
                proposal = await self._generate_proposal(proposer, task, context)
                current_proposal = proposal

                # Record the proposal in the history
                self._record_interaction(
                    agent_id=proposer.id,
                    action="propose",
                    content=proposal,
                    round=round_num,
                    trace_id=trace_id,
                )

                logger.info(f"Initial proposal from {proposer.name}: {proposal[:100]}...")

            # Collect feedback from all other agents
            feedback = []
            for agent in debate_agents[1:] if round_num == 1 else debate_agents:
                # Skip the proposer in the first round
                if round_num == 1 and agent.id == debate_agents[0].id:
                    continue

                # Generate feedback from this agent
                agent_feedback = await self._generate_feedback(
                    agent, task, context, current_proposal
                )
                feedback.append((agent.id, agent_feedback))

                # Record the feedback in the history
                self._record_interaction(
                    agent_id=agent.id,
                    action="feedback",
                    content=agent_feedback,
                    round=round_num,
                    trace_id=trace_id,
                )

                logger.info(f"Feedback from {agent.name}: {agent_feedback[:100]}...")

            # Check if we've reached consensus
            consensus_score = await self._evaluate_consensus(
                task, context, current_proposal, feedback
            )

            logger.info(f"Consensus score: {consensus_score:.2f}")

            if consensus_score >= self.config.consensus_threshold:
                consensus_reached = True
                logger.info(f"Consensus reached in round {round_num}")
                break

            # If no consensus, refine the proposal
            if round_num < max_rounds:
                # Choose a different agent to make the next proposal
                next_proposer = debate_agents[round_num % len(debate_agents)]

                # Generate a refined proposal
                refined_proposal = await self._generate_refined_proposal(
                    next_proposer, task, context, current_proposal, feedback
                )
                current_proposal = refined_proposal

                # Record the refined proposal in the history
                self._record_interaction(
                    agent_id=next_proposer.id,
                    action="refine",
                    content=refined_proposal,
                    round=round_num,
                    trace_id=trace_id,
                )

                logger.info(
                    f"Refined proposal from {next_proposer.name}: {refined_proposal[:100]}..."
                )

        # Return the final proposal
        if consensus_reached:
            return f"Consensus reached: {current_proposal}"
        return f"Max rounds reached without consensus. Final proposal: {current_proposal}"

    async def _run_contract_net(
        self,
        task: str,
        context: str | None,
        max_rounds: int,
        trace_id: str | None = None,
    ) -> str:
        """
        Run a contract-net protocol for task delegation.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            max_rounds: Maximum number of rounds

        Returns:
        -------
            str: Result of the contract-net protocol

        """
        # Identify the manager agent (first agent by default)
        if not self.agents:
            msg = "Contract-net requires at least one agent"
            raise ValueError(msg)

        manager_id = next(iter(self.agents.keys()))
        manager = self.agents[manager_id]

        # Identify worker agents (all agents except the manager)
        workers = {
            agent_id: agent for agent_id, agent in self.agents.items() if agent_id != manager_id
        }

        if not workers:
            msg = "Contract-net requires at least one worker agent"
            raise ValueError(msg)

        logger.info(f"Starting contract-net with manager {manager.name} and {len(workers)} workers")

        # Step 1: Task announcement
        subtasks = await self._generate_subtasks(manager, task, context)

        # Record the task announcement in the history
        self._record_interaction(
            agent_id=manager_id,
            action="announce",
            content=json.dumps(subtasks),
            round=1,
            trace_id=trace_id,
        )

        logger.info(f"Manager announced {len(subtasks)} subtasks")

        # Step 2: Bid submission
        bids = {}
        for worker_id, worker in workers.items():
            # Generate bids for each subtask
            worker_bids = await self._generate_bids(worker, subtasks, context)
            bids[worker_id] = worker_bids

            # Record the bids in the history
            self._record_interaction(
                agent_id=worker_id,
                action="bid",
                content=json.dumps(worker_bids),
                round=1,
                trace_id=trace_id,
            )

            logger.info(f"Worker {worker.name} submitted bids for {len(worker_bids)} subtasks")

        # Step 3: Bid evaluation and task allocation
        allocations = await self._evaluate_bids(manager, subtasks, bids)

        # Record the allocations in the history
        self._record_interaction(
            agent_id=manager_id,
            action="allocate",
            content=json.dumps(allocations),
            round=1,
            trace_id=trace_id,
        )

        logger.info(f"Manager allocated {len(allocations)} subtasks")

        # Step 4: Task execution
        results = {}
        for subtask_id, worker_id in allocations.items():
            # Find the subtask and worker
            subtask = next(st for st in subtasks if st["id"] == subtask_id)
            worker = workers[worker_id]

            # Execute the subtask
            result = await self._execute_subtask(worker, subtask, context)
            results[subtask_id] = result

            # Record the result in the history
            self._record_interaction(
                agent_id=worker_id,
                action="execute",
                content=result,
                round=2,
                trace_id=trace_id,
            )

            logger.info(f"Worker {worker.name} executed subtask {subtask_id}")

        # Step 5: Result aggregation
        final_result = await self._aggregate_results(manager, task, subtasks, results)

        # Record the final result in the history
        self._record_interaction(
            agent_id=manager_id,
            action="aggregate",
            content=final_result,
            round=2,
            trace_id=trace_id,
        )

        logger.info(f"Manager aggregated results: {final_result[:100]}...")

        return f"Task completed: {final_result}"

    async def _generate_proposal(
        self,
        agent: AgentNode,
        task: str,
        context: str | None,
    ) -> str:
        """
        Generate a proposal from an agent.

        Args:
        ----
            agent: Agent to generate the proposal
            task: Task to solve
            context: Additional context for the task

        Returns:
        -------
            str: Generated proposal

        """
        # Create the prompt
        prompt = f"""
        You are {agent.name}, a {agent.role}. {agent.description}

        Task: {task}

        {f"Context: {context}" if context else ""}

        Generate a detailed proposal to solve this task. Be specific and thorough.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(agent)

        # Generate the proposal
        response = await model.generate(prompt=prompt.strip())

        # Ensure we return a string even if response.text is None
        return response.text or ""

    def _get_agent_model(self, agent: AgentNode) -> LLM:
        """
        Get the model to use for an agent.

        If the agent has a base Agent, use its model.
        If the agent has a provider and model specified, create a model for it.
        Otherwise, use the default model.

        Args:
        ----
            agent: Agent to get the model for

        Returns:
        -------
            LLM: Model to use for the agent

        """
        # If the agent has a base Agent, use its model
        if agent.agent:
            return agent.agent.model

        # If the agent has a provider and model specified, create a model for it
        if agent.provider and agent.model:
            # Create a model using the new approach
            return LLM.create(provider=agent.provider, model=agent.model, **agent.model_parameters)

        # Otherwise, use the default model
        return self.model

    async def _generate_feedback(
        self,
        agent: AgentNode,
        task: str,
        context: str | None,
        proposal: str,
    ) -> str:
        """
        Generate feedback on a proposal from an agent.

        Args:
        ----
            agent: Agent to generate the feedback
            task: Task to solve
            context: Additional context for the task
            proposal: Proposal to provide feedback on

        Returns:
        -------
            str: Generated feedback

        """
        # Create the prompt
        prompt = f"""
        You are {agent.name}, a {agent.role}. {agent.description}

        Task: {task}

        {f"Context: {context}" if context else ""}

        Proposal:
        {proposal}

        Provide constructive feedback on this proposal. Identify strengths and weaknesses,
        and suggest specific improvements. Be detailed and helpful.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(agent)

        # Generate the feedback
        response = await model.generate(prompt=prompt.strip())

        # Ensure we return a string even if response.text is None
        return response.text or ""

    async def _evaluate_consensus(
        self,
        task: str,
        context: str | None,
        proposal: str,
        feedback: list[tuple[str, str]],
    ) -> float:
        """
        Evaluate the level of consensus on a proposal.

        Args:
        ----
            task: Task to solve
            context: Additional context for the task
            proposal: Current proposal
            feedback: List of (agent_id, feedback) tuples

        Returns:
        -------
            float: Consensus score (0.0 to 1.0)

        """
        # Create the prompt
        prompt = f"""
        Task: {task}

        {f"Context: {context}" if context else ""}

        Proposal:
        {proposal}

        Feedback from agents:
        {chr(10).join([f"Agent {agent_id}: {feedback[:200]}..." for agent_id, feedback in feedback])}

        Evaluate the level of consensus among the agents on this proposal.
        Consider how much agreement there is and how significant the disagreements are.

        Return a consensus score between 0.0 (no consensus) and 1.0 (full consensus).
        Provide only the numeric score, with no additional text.
        """

        # Generate the evaluation
        response = await self.model.generate(prompt=prompt.strip())

        # Extract the score
        try:
            # Ensure response.text is not None
            text = response.text or ""
            score = float(text.strip())
            # Ensure the score is between 0.0 and 1.0
            return max(0.0, min(1.0, score))
        except ValueError:
            logger.warning(f"Failed to parse consensus score: {response.text}")
            return 0.0

    async def _generate_refined_proposal(
        self,
        agent: AgentNode,
        task: str,
        context: str | None,
        current_proposal: str,
        feedback: list[tuple[str, str]],
    ) -> str:
        """
        Generate a refined proposal based on feedback.

        Args:
        ----
            agent: Agent to generate the refined proposal
            task: Task to solve
            context: Additional context for the task
            current_proposal: Current proposal
            feedback: List of (agent_id, feedback) tuples

        Returns:
        -------
            str: Refined proposal

        """
        # Create the prompt
        prompt = f"""
        You are {agent.name}, a {agent.role}. {agent.description}

        Task: {task}

        {f"Context: {context}" if context else ""}

        Current proposal:
        {current_proposal}

        Feedback from agents:
        {chr(10).join([f"Agent {agent_id}: {feedback}" for agent_id, feedback in feedback])}

        Based on the feedback, generate a refined proposal that addresses the concerns
        and incorporates the suggestions. Be specific and thorough.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(agent)

        # Generate the refined proposal
        response = await model.generate(prompt=prompt.strip())

        # Ensure we return a string even if response.text is None
        return response.text or ""

    async def _generate_subtasks(
        self,
        manager: AgentNode,
        task: str,
        context: str | None,
    ) -> list[dict[str, Any]]:
        """
        Generate subtasks for a task.

        Args:
        ----
            manager: Manager agent
            task: Task to solve
            context: Additional context for the task

        Returns:
        -------
            List[Dict[str, Any]]: List of subtasks

        """
        # Create the prompt
        prompt = f"""
        You are {manager.name}, a {manager.role}. {manager.description}

        Task: {task}

        {f"Context: {context}" if context else ""}

        Break down this task into smaller subtasks that can be delegated to other agents.
        For each subtask, provide:
        1. A unique ID (e.g., "subtask1")
        2. A clear description of what needs to be done
        3. Any specific requirements or constraints

        Format your response as a JSON array of subtask objects, each with "id", "description",
        and "requirements" fields.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(manager)

        # Generate the subtasks
        response = await model.generate(prompt=prompt.strip())

        # Parse the JSON response
        try:
            # Ensure response.text is not None
            text = response.text or ""

            # Extract JSON from the response if it's wrapped in ```json ... ```
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                json_str = text.split("```json", 1)[1].split("```", 1)[0]
                subtasks = json.loads(json_str)
            else:
                # Try to parse the whole response as JSON
                subtasks = json.loads(text)

            # Ensure it's a list
            if not isinstance(subtasks, list):
                msg = "Subtasks must be a list"
                raise ValueError(msg)

            return subtasks
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse subtasks: {e}")
            # Fall back to a simple subtask
            return [
                {
                    "id": "subtask1",
                    "description": task,
                    "requirements": "Complete the task as described",
                }
            ]

    async def _generate_bids(
        self,
        worker: AgentNode,
        subtasks: list[dict[str, Any]],
        context: str | None,
    ) -> dict[str, float]:
        """
        Generate bids for subtasks from a worker agent.

        Args:
        ----
            worker: Worker agent
            subtasks: List of subtasks
            context: Additional context for the task

        Returns:
        -------
            Dict[str, float]: Dictionary of subtask ID to bid value

        """
        # Create the prompt
        prompt = f"""
        You are {worker.name}, a {worker.role}. {worker.description}

        {f"Context: {context}" if context else ""}

        Available subtasks:
        {json.dumps(subtasks, indent=2)}

        For each subtask, provide a bid representing your confidence in completing it successfully.
        The bid should be a value between 0.0 (cannot complete) and 1.0 (can complete perfectly).

        Format your response as a JSON object mapping subtask IDs to bid values.
        For example: {{"subtask1": 0.8, "subtask2": 0.6}}
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(worker)

        # Generate the bids
        response = await model.generate(prompt=prompt.strip())

        # Parse the JSON response
        try:
            # Ensure response.text is not None
            text = response.text or ""

            # Extract JSON from the response if it's wrapped in ```json ... ```
            if "```json" in text and "```" in text.split("```json", 1)[1]:
                json_str = text.split("```json", 1)[1].split("```", 1)[0]
                bids = json.loads(json_str)
            else:
                # Try to parse the whole response as JSON
                bids = json.loads(text)

            # Ensure it's a dictionary
            if not isinstance(bids, dict):
                msg = "Bids must be a dictionary"
                raise ValueError(msg)

            # Ensure all values are between 0.0 and 1.0
            for subtask_id, bid in bids.items():
                bids[subtask_id] = max(0.0, min(1.0, float(bid)))

            return bids
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse bids: {e}")
            # Fall back to a simple bid
            return {subtask["id"]: 0.5 for subtask in subtasks}

    async def _evaluate_bids(
        self,
        manager: AgentNode,
        subtasks: list[dict[str, Any]],
        bids: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        """
        Evaluate bids and allocate subtasks to workers.

        Args:
        ----
            manager: Manager agent
            subtasks: List of subtasks
            bids: Dictionary of worker ID to dictionary of subtask ID to bid value

        Returns:
        -------
            Dict[str, str]: Dictionary of subtask ID to worker ID

        """
        # Simple allocation strategy: assign each subtask to the worker with the highest bid
        allocations = {}

        for subtask in subtasks:
            subtask_id = subtask["id"]
            best_worker_id = None
            best_bid = -1.0

            for worker_id, worker_bids in bids.items():
                if subtask_id in worker_bids and worker_bids[subtask_id] > best_bid:
                    best_worker_id = worker_id
                    best_bid = worker_bids[subtask_id]

            if best_worker_id is not None:
                allocations[subtask_id] = best_worker_id

        return allocations

    async def _execute_subtask(
        self,
        worker: AgentNode,
        subtask: dict[str, Any],
        context: str | None,
    ) -> str:
        """
        Execute a subtask with a worker agent.

        Args:
        ----
            worker: Worker agent
            subtask: Subtask to execute
            context: Additional context for the task

        Returns:
        -------
            str: Result of the subtask execution

        """
        # Create the prompt
        prompt = f"""
        You are {worker.name}, a {worker.role}. {worker.description}

        {f"Context: {context}" if context else ""}

        Subtask: {subtask["description"]}
        Requirements: {subtask.get("requirements", "None")}

        Execute this subtask and provide a detailed result. Be specific and thorough.
        """

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(worker)

        # Generate the result
        response = await model.generate(prompt=prompt.strip())

        # Ensure we return a string even if response.text is None
        return response.text or ""

    async def _aggregate_results(
        self,
        manager: AgentNode,
        task: str,
        subtasks: list[dict[str, Any]],
        results: dict[str, str],
    ) -> str:
        """
        Aggregate results from subtasks.

        Args:
        ----
            manager: Manager agent
            task: Original task
            subtasks: List of subtasks
            results: Dictionary of subtask ID to result

        Returns:
        -------
            str: Aggregated result

        """
        # Create the prompt
        prompt = f"""
        You are {manager.name}, a {manager.role}. {manager.description}

        Original task: {task}

        Subtasks and results:
        """

        # Add each subtask and its result
        for subtask in subtasks:
            subtask_id = subtask["id"]
            if subtask_id in results:
                prompt += f"\n\nSubtask: {subtask['description']}\nResult: {results[subtask_id]}"

        prompt += (
            "\n\nAggregate these results into a cohesive final solution for the original task."
        )

        # Use agent-specific model if available, otherwise use the default model
        model = self._get_agent_model(manager)

        # Generate the aggregated result
        response = await model.generate(prompt=prompt.strip())

        # Ensure we return a string even if response.text is None
        return response.text or ""

    def _record_interaction(
        self,
        agent_id: str,
        action: str,
        content: str,
        round: int,
        trace_id: str | None = None,
    ) -> None:
        """
        Record an agent interaction in the history.

        Args:
        ----
            agent_id: ID of the agent
            action: Action performed
            content: Content of the interaction
            round: Round number
            trace_id: ID of the trace for monitoring

        """
        if not self.config.logging_enabled:
            return

        interaction = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "action": action,
            "content": content,
            "round": round,
        }

        self.history.append(interaction)

        # Add a span to the trace if monitoring is enabled
        if self.trace_manager and trace_id:
            self.add_span(trace_id, agent_id, action, content)

    def get_history(self):
        """
        Get the interaction history.

        Returns
        -------
            List[Dict[str, Any]]: List of interaction records

        """
        return self.history

    def clear_history(self):
        """Clear the interaction history."""
        self.history = []
