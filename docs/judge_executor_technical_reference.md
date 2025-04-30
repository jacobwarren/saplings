# JudgeAgent and Executor Technical Reference

This technical reference provides detailed information about the implementation of the JudgeAgent and Executor components in the Saplings framework.

## JudgeAgent Implementation

### Class Structure

The JudgeAgent module consists of several key classes:

1. `JudgeAgent`: The main class for evaluating outputs
2. `JudgeResult`: Represents the result of a judgment
3. `DimensionScore`: Represents a score for a single dimension
4. `JudgeConfig`: Configuration for the JudgeAgent
5. `Rubric`: Defines evaluation criteria
6. `RubricItem`: Represents a single item in a rubric
7. `ScoringDimension`: Enum of dimensions for scoring outputs
8. `CritiqueFormat`: Enum of formats for critique output

### Key Methods

#### JudgeAgent._create_judgment_prompt()

```python
def _create_judgment_prompt(
    self, output: str, prompt: str, rubric: Optional[Rubric] = None
) -> str:
```

This method creates a prompt for the LLM to judge an output. The prompt includes:

1. Instructions for the LLM to act as a judge
2. The original prompt
3. The output to judge
4. The evaluation criteria from the rubric
5. Instructions for the response format

Example prompt structure:
```
You are a judge evaluating the quality of an AI assistant's response.
Your task is to provide a fair and detailed assessment based on the given criteria.

# Original Prompt
{prompt}

# AI Assistant's Response
{output}

# Evaluation Criteria
## Relevance
How relevant the output is to the prompt
Score levels:
- 0.0: Completely irrelevant
- 0.5: Somewhat relevant
- 1.0: Highly relevant

...

# Instructions
Please evaluate the response according to the criteria above. For each dimension:
1. Assign a score between 0.0 and 1.0
2. Provide a brief explanation for your score
3. Calculate an overall score as a weighted average of the dimension scores
4. Provide a detailed critique of the response
5. Suggest specific improvements

# Response Format
...
```

#### JudgeAgent._parse_judgment_response()

```python
def _parse_judgment_response(self, response: str) -> Dict[str, Any]:
```

This method parses the LLM's judgment response into a structured format. It handles different response formats:

1. **JSON**: Extracts JSON from the response
2. **Structured/Markdown**: Parses sections like "DIMENSION SCORES", "OVERALL SCORE", "CRITIQUE", and "SUGGESTIONS"

The method returns a dictionary with:
- `dimension_scores`: List of scores for individual dimensions
- `overall_score`: Overall score (0.0 to 1.0)
- `critique`: Detailed critique of the output
- `suggestions`: List of suggestions for improvement

#### JudgeAgent._calculate_overall_score()

```python
def _calculate_overall_score(
    self, dimension_scores: List[Dict[str, Any]], rubric: Optional[Rubric] = None
) -> float:
```

This method calculates the overall score from dimension scores using a weighted average:

1. Creates a mapping of dimension to weight from the rubric
2. Calculates the weighted sum of scores
3. Divides by the total weight to get the weighted average

#### JudgeAgent.judge()

```python
async def judge(
    self, output: str, prompt: str, rubric: Optional[Rubric] = None
) -> JudgeResult:
```

This is the main method for judging an output:

1. Creates a judgment prompt using `_create_judgment_prompt()`
2. Sends the prompt to the LLM using `model.generate()`
3. Parses the response using `_parse_judgment_response()`
4. Calculates the overall score if not provided using `_calculate_overall_score()`
5. Determines if the output passed verification based on the threshold
6. Creates dimension scores from the parsed judgment
7. Creates and returns a JudgeResult
8. Updates statistics (total judgments, passed judgments, tokens, cost)

#### JudgeAgent.format_critique()

```python
def format_critique(self, result: JudgeResult) -> str:
```

This method formats a JudgeResult for display based on the configured format:

1. **JSON**: Returns a JSON representation of the result
2. **Markdown**: Returns a markdown-formatted critique with headers and sections
3. **Structured**: Returns a plain text critique with structured sections
4. **Simple**: Returns a minimal plain text critique

## Executor Implementation

### Verification Integration

The Executor integrates with the JudgeAgent through the `_verify_output()` method:

```python
async def _verify_output(
    self, output: str, prompt: str, verification_strategy: Optional[VerificationStrategy] = None, **kwargs
) -> Tuple[bool, Optional[float], Optional[str]]:
```

For the `JUDGE` strategy, the implementation is:

```python
# JudgeAgent verification
if strategy == VerificationStrategy.JUDGE:
    if self.judge_agent is None:
        logger.warning("JudgeAgent not provided, falling back to basic verification")
        return await self._verify_output(
            output=output,
            prompt=prompt,
            verification_strategy=VerificationStrategy.BASIC,
            **kwargs,
        )

    # Use JudgeAgent to verify the output
    try:
        # Judge the output using JudgeAgent
        judge_result = await self.judge_agent.judge(output=output, prompt=prompt)

        # Get the verification result
        passed = judge_result.passed
        score = judge_result.overall_score

        # Format the feedback
        feedback = self.judge_agent.format_critique(judge_result)

        # Log the verification result
        if passed:
            logger.info(f"Output passed verification with score {score:.2f}")
        else:
            logger.warning(f"Output failed verification with score {score:.2f}")

        return passed, score, feedback
    except Exception as e:
        logger.error(f"Error during JudgeAgent verification: {e}")
        logger.warning("Falling back to basic verification")
        return await self._verify_output(
            output=output,
            prompt=prompt,
            verification_strategy=VerificationStrategy.BASIC,
            **kwargs,
        )
```

This implementation:
1. Checks if a JudgeAgent is provided, falling back to basic verification if not
2. Calls the JudgeAgent's `judge()` method to evaluate the output
3. Extracts the verification result (passed, score) from the JudgeResult
4. Formats the feedback using the JudgeAgent's `format_critique()` method
5. Logs the verification result
6. Returns the verification result (passed, score, feedback)
7. Handles exceptions by falling back to basic verification

### Refinement Implementation

The Executor implements refinement through the `_refine_output()` method:

```python
async def _refine_output(
    self,
    prompt: str,
    rejected_output: str,
    verification_feedback: Optional[str],
    documents: Optional[List[Document]] = None,
    attempt: int = 1,
    **kwargs,
) -> LLMResponse:
```

For the `FEEDBACK` strategy, the implementation is:

```python
# Feedback refinement
if strategy == RefinementStrategy.FEEDBACK:
    # Create a new prompt with feedback
    feedback_prompt = (
        f"{prompt}\n\n"
        f"Previous attempt was rejected for the following reason:\n"
        f"{verification_feedback or 'Unknown reason'}\n\n"
        f"Please try again, addressing the feedback. "
        f"This is refinement attempt #{attempt} out of {self.config.max_refinement_attempts}."
    )

    # Generate with the feedback prompt
    return await self._generate_final(feedback_prompt, documents, **kwargs)
```

For the `ITERATIVE` strategy, the implementation is:

```python
# Iterative refinement
if strategy == RefinementStrategy.ITERATIVE:
    # Create a new prompt with feedback and the rejected output
    iterative_prompt = (
        f"{prompt}\n\n"
        f"Previous attempt (#{attempt}):\n{rejected_output}\n\n"
        f"Feedback:\n{verification_feedback or 'Unknown reason'}\n\n"
        f"Please improve the previous attempt based on the feedback. "
        f"This is refinement attempt #{attempt} out of {self.config.max_refinement_attempts}."
    )

    # Generate with the iterative prompt
    return await self._generate_final(iterative_prompt, documents, **kwargs)
```

### Execution Flow

The execution flow in the `execute()` method integrates verification and refinement:

```python
# Verify the output
verified, verification_score, verification_feedback = await self._verify_output(
    output=final_response.text,
    prompt=prompt,
    **kwargs,
)

# Refine the output if verification failed
refinement_attempts = 0
while (
    not verified
    and refinement_attempts < self.config.max_refinement_attempts
    and self.config.refinement_strategy != RefinementStrategy.NONE
):
    refinement_attempts += 1
    logger.info(
        f"Output verification failed (score: {verification_score}). "
        f"Refinement attempt {refinement_attempts}/{self.config.max_refinement_attempts}"
    )

    # Refine the output
    refined_response = await self._refine_output(
        prompt=prompt,
        rejected_output=final_response.text,
        verification_feedback=verification_feedback,
        documents=documents,
        attempt=refinement_attempts,
        **kwargs,
    )

    # Update final response
    final_response = refined_response

    # Verify the refined output
    verified, verification_score, verification_feedback = await self._verify_output(
        output=final_response.text,
        prompt=prompt,
        **kwargs,
    )
```

This flow:
1. Verifies the output using `_verify_output()`
2. If verification fails, enters a refinement loop
3. Refines the output using `_refine_output()`
4. Verifies the refined output
5. Repeats until verification passes or max attempts are reached

## Streaming Integration

The JudgeAgent and Executor also integrate with streaming output:

```python
async def _execute_streaming(
    self,
    prompt: str,
    documents: Optional[List[Document]] = None,
    on_draft: Optional[Callable[[str], None]] = None,
    on_chunk: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> ExecutionResult:
```

The streaming implementation:
1. Generates a draft response if speculative execution is enabled
2. Generates a final response with streaming
3. Verifies the output
4. Refines the output if verification fails
5. Calls the chunk callback for each refinement

```python
# Verify the output
verified, verification_score, verification_feedback = await self._verify_output(
    output=final_text,
    prompt=prompt,
    **kwargs,
)

# Refine the output if verification failed
refinement_attempts = 0
while (
    not verified
    and refinement_attempts < self.config.max_refinement_attempts
    and self.config.refinement_strategy != RefinementStrategy.NONE
):
    refinement_attempts += 1
    logger.info(
        f"Output verification failed (score: {verification_score}). "
        f"Refinement attempt {refinement_attempts}/{self.config.max_refinement_attempts}"
    )

    # Refine the output
    refined_response = await self._refine_output(
        prompt=prompt,
        rejected_output=final_text,
        verification_feedback=verification_feedback,
        documents=documents,
        attempt=refinement_attempts,
        **kwargs,
    )

    # Update final response
    final_response = refined_response
    final_text = refined_response.text

    # Call chunk callback if provided
    if on_chunk is not None:
        on_chunk(f"\n[Refinement {refinement_attempts}]\n{final_text}")

    # Verify the refined output
    verified, verification_score, verification_feedback = await self._verify_output(
        output=final_text,
        prompt=prompt,
        **kwargs,
    )
```

## Error Handling

The integration includes robust error handling:

1. **JudgeAgent Availability**: Falls back to basic verification if JudgeAgent is not provided
2. **Verification Errors**: Catches exceptions during verification and falls back to basic verification
3. **Parsing Errors**: Handles errors when parsing judgment responses
4. **Refinement Limits**: Enforces a maximum number of refinement attempts

## Performance Considerations

The implementation includes several performance optimizations:

1. **Caching**: Caches execution results to avoid redundant generation
2. **Latency Tracking**: Tracks latency for draft generation, final generation, and total execution
3. **Budget Enforcement**: Enforces budget constraints for judgments
4. **Token Limits**: Respects token limits for judgment prompts

## Testing

The integration is thoroughly tested:

1. **Unit Tests**: Tests for individual components (JudgeAgent, Executor)
2. **Integration Tests**: Tests for the integration between JudgeAgent and Executor
3. **Mock Models**: Uses mock LLMs for testing
4. **Edge Cases**: Tests for error handling and fallback mechanisms

## Future Enhancements

Planned enhancements for the integration:

1. **ValidatorRegistry**: Integration with a registry of validators for specific verification tasks
2. **Multi-Model Verification**: Using multiple models for verification
3. **Adaptive Refinement**: Dynamically adjusting refinement strategies based on verification results
4. **Feedback Aggregation**: Aggregating feedback from multiple verification sources
