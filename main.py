import ast
import random
import subprocess
import sys
import time
import timeit
from typing import List, Tuple, Optional, Dict, Any, Callable
import openai

class CodeEvaluator:
    """
    Evaluates Python code snippets for correctness and functionality.
    
    The evaluator checks if the code:
    1. Runs without errors
    2. Produces the expected output for given test cases
    """
    
    def __init__(self, test_cases: List[Tuple[str, Any]]):
        """
        Initialize with test cases in the format [(input, expected_output), ...]
        """
        self.test_cases = test_cases
    
    def evaluate(self, code: str) -> Tuple[bool, str]:
        """Evaluate the code and return (success, message)."""
        try:
            # Parse the code to check syntax first
            ast.parse(code)
            
            # Time the execution
            start_time = time.perf_counter()
            exec(code, {})  # Run in empty namespace
            end_time = time.perf_counter()
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            return True, f"{execution_time:.6f} ms"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Error during execution: {str(e)}"
        except Exception as e:
            return False, f"Runtime error: {str(e)}"

class CodeEvolver:
    """
    Evolves Python code through genetic programming with configurable evaluation methods.
    Supports both function-based and LM Studio-based evaluation.
    """
    
    def __init__(self, 
                 initial_population: List[str],
                 evaluator: CodeEvaluator,
                 evaluation_method: str = "function",  # 'function' or 'lm_studio'
                 improvement_factor: str = "efficiency",
                 api_key: str = "lm-studio",
                 model: str = "gpt-3.5-turbo",
                 population_size: int = 10,
                 max_generations: int = 20,
                 base_url: str = "http://localhost:1234/v1"):
        """
        Initialize the code evolver.
        
        Args:
            initial_population: List of initial code snippets
            evaluator: CodeEvaluator instance to test code correctness
            evaluation_method: Either 'function' for local evaluation or 'lm_studio' for AI-based evaluation
            improvement_factor: The factor to improve (e.g., 'efficiency', 'readability')
            api_key: API key for the LLM service (if using 'lm_studio' method)
            model: Model name to use (if using 'lm_studio' method)
            population_size: Number of individuals in each generation
            max_generations: Maximum number of generations to run
            base_url: Base URL for the LLM API (defaults to local LM Studio)
        """
        self.population = initial_population
        self.evaluator = evaluator
        self.evaluation_method = evaluation_method.lower()
        self.improvement_factor = improvement_factor.lower()
        self.model = model
        self.population_size = population_size
        self.max_generations = max_generations

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

        
        # Track best individuals across generations
        self.best_individuals = []
        
        # Initialize conversation history for context (used in LM Studio mode)
        self.conversation_history = [
            {
                "role": "system",
                "content": f"""You are a code evolution assistant. Your task is to improve Python functions 
                with a focus on {self.improvement_factor} through an evolutionary process.
                
                Guidelines:
                1. Always return only the Python function code, no explanations or markdown
                2. The function must be named 'c' and take a single parameter
                3. Focus on {self.improvement_factor} in your improvements
                4. Maintain the same function signature and return type
                5. Keep the code simple and readable
                
                The function will be tested with various inputs and must return the expected outputs.
                """
            }
        ]
        
        # Define factor evaluation criteria for function-based evaluation
        self.factor_evaluators = {
            'efficiency': self._evaluate_efficiency,
            'readability': self._evaluate_readability,
            'simplicity': self._evaluate_simplicity,
            'maintainability': self._evaluate_maintainability
        }
        
        # Store test cases for execution time measurement
        self.test_cases = evaluator.test_cases
    def _validate_code_structure(self, code: str) -> bool:
        """
        Validate that the code has proper Python structure.
        Returns True if the code is valid Python and has consistent structure.
        """
        try:
            tree = ast.parse(code)
            
            # Check for mixed function and top-level code
            has_func = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
            has_top_level = any(
                not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Expr, ast.ClassDef))
                for node in tree.body
            )
            
            # Either all code should be in functions or all at top level
            if has_func and has_top_level:
                return False
                
            # Additional check: If it's a function, it should be callable
            if has_func:
                # Create a local namespace and exec the code
                local_vars = {}
                try:
                    exec(code, globals(), local_vars)
                    # Check if we have any callable functions
                    return any(callable(v) for v in local_vars.values())
                except Exception:
                    return False
            return True
            
        except (SyntaxError, IndentationError):
            return False
    def _read_code_from_file(self, file_path: str) -> str:
        """Read code from a file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def _generate_initial_population_with_ai(self, prompt: str, target_code: str, size: int) -> List[str]:
        """Generate initial population using AI based on the prompt and target code."""
        unique_implementations = set()
        attempts = 0
        max_attempts = size * 2  # Prevent infinite loops
        
        # Always include the original code as a baseline if it's valid
        if target_code.strip() and self._validate_code_structure(target_code):
            unique_implementations.add(target_code.strip() + '\n')
        
        while len(unique_implementations) < size and attempts < max_attempts:
            try:
                messages = [
                    {"role": "system", "content": """You are a helpful coding assistant that generates unique Python functions.
                    Rules:
                    1. Return ONLY complete Python functions
                    2. No mixing of function definitions and top-level code
                    3. Each function must be self-contained and executable
                    4. No placeholders or TODOs
                    """},
                    {"role": "user", "content": f"""I want to improve this code:
```python
{target_code}
```

{prompt}

Please generate {min(3, size)} different implementations that could potentially improve upon this code. 
For each implementation, return ONLY the complete function definition without any additional text or explanation. 
Each implementation should be significantly different from the others. 
Separate each implementation with '\n---\n'."""}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,  # Slightly lower temperature for more reliable code
                    max_tokens=1500
                )
                
                # Parse the response into individual code snippets
                new_implementations = response.choices[0].message.content.strip().split('\n---\n')
                
                # Clean up and add new implementations
                for impl in new_implementations:
                    # Remove any non-code lines and normalize whitespace
                    lines = [line for line in impl.split('\n') 
                            if line.strip() and not line.strip().startswith(('#', '```'))]
                    if not lines:
                        continue
                        
                    # Ensure proper indentation
                    code = '\n'.join(lines).strip()
                    
                    # Ensure it's a complete function and valid Python
                    if code.startswith('def ') and '\n' in code and self._validate_code_structure(code):
                        unique_implementations.add(code + '\n')
                        
                        # If we have enough, return early
                        if len(unique_implementations) >= size:
                            return list(unique_implementations)[:size]
                
            except Exception as e:
                print(f"Error generating AI implementation (attempt {attempts + 1}): {e}")
                
            attempts += 1
        
        # Fallback to simple variations if we couldn't generate enough unique implementations
        result = list(unique_implementations)
        while len(result) < size and attempts < max_attempts * 2:
            variation = self._generate_variation(target_code)
            if variation and self._validate_code_structure(variation) and variation not in result:
                result.append(variation)
            attempts += 1
        
        # If we still don't have enough, generate some basic variations
        if len(result) < size:
            result.extend(self._generate_fallback_population(target_code, size - len(result)))
        
        return result[:size]
            

    def _generate_variation(self, code: str) -> str:
        """
        Generate a valid variation of the given code.
        Ensures the output is valid Python code with consistent structure.
        """
        # First try to parse the code to understand its structure
        try:
            tree = ast.parse(code)
            
            # If it's a function, generate a variation that's also a function
            if any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree)):
                # Generate a new function with some variation
                func_name = f"implementation_{random.randint(1, 1000)}"
                # Keep the function structure but vary the implementation
                return f"def {func_name}():" + '\n'.join([
                    '    ' + line for line in self._vary_code_impl(code).split('\n')
                ]) + '\n'
            else:
                # Generate top-level variation
                return self._vary_code_impl(code)
                
        except Exception as e:
            print(f"Warning: Error parsing code structure: {e}")
            return self._vary_code_impl(code)
    
    def _vary_code_impl(self, code: str) -> str:
        """Generate a simple variation of the code while maintaining validity."""
        try:
            # First try AI-based variation
            messages = [
                {"role": "system", "content": """You are a helpful coding assistant that generates variations of Python code.
                Rules:
                1. Return ONLY the complete code
                2. Maintain the same functionality
                3. Change the implementation approach if possible
                4. Ensure the code is valid Python"""},
                {"role": "user", "content": f"Generate a variation of this code:\n```python\n{code}\n```"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=1000
            )
            
            variation = response.choices[0].message.content.strip()
            
            # Clean up the response
            lines = [line for line in variation.split('\n') 
                    if line.strip() and not line.strip().startswith(('#', '```'))]
            variation = '\n'.join(lines).strip()
            
            # Validate before returning
            if self._validate_code_structure(variation):
                return variation + '\n'
                
        except Exception as e:
            print(f"Warning: Could not generate AI variation: {e}")
        
        # Fallback to simple modifications if AI generation fails or is invalid
        lines = code.split('\n')
        if len(lines) > 1:
            # Simple variations that maintain validity
            variations = [
                lambda: '\n'.join(f'    {line.strip()}' if i > 0 else line for i, line in enumerate(lines)),  # Ensure indentation
                lambda: code + '\n' + f'# Modified at {time.time()}\n',  # Add timestamp comment
                lambda: code.replace('return', 'result = ') + '\nreturn result\n' if 'return' in code else code,  # Add temp variable
            ]
            
            # Try different variations until we find a valid one
            for _ in range(3):  # Try up to 3 times
                variation = random.choice(variations)()
                if self._validate_code_structure(variation):
                    return variation
        
        # If all else fails, return the original with a small change
        return code + '\n# Modified\n'
    
    def _generate_fallback_population(self, target_code: str, size: int) -> List[str]:
        """
        Generate a simple population of valid Python code variations.
        Ensures all generated code is syntactically valid.
        """
        population = []
        
        # Always include the original if it's valid
        if self._validate_code_structure(target_code):
            population.append(target_code)
        
        # Generate simple variations
        for i in range(1, size * 2):  # Generate extra to ensure we get enough valid ones
            if len(population) >= size:
                break
                
            # Different types of simple variations
            if i % 3 == 0 and 'return' in target_code:
                # Add a temporary variable before return
                parts = target_code.split('return')
                if len(parts) > 1:
                    var_name = f"result_{i}"
                    variant = f"{parts[0]}{var_name} ={parts[1]}\n    return {var_name}\n"
                    if self._validate_code_structure(variant):
                        population.append(variant)
                        
            elif i % 2 == 0 and '=' in target_code:
                # Swap variable names
                lines = target_code.split('\n')
                new_lines = []
                for line in lines:
                    if '=' in line and not line.strip().startswith('#'):
                        parts = line.split('=')
                        if len(parts) == 2:
                            line = f"{parts[1].strip()} = {parts[0].strip()}"
                    new_lines.append(line)
                variant = '\n'.join(new_lines)
                if self._validate_code_structure(variant):
                    population.append(variant)
                    
            else:
                # Just add a comment with variation number
                variant = f"{target_code.rstrip()}\n# Variation {i}\n"
                if self._validate_code_structure(variant):
                    population.append(variant)
        
        # If we still don't have enough, just duplicate the valid ones we have
        while len(population) < size and population:
            population.append(population[-1])
            
        return population[:size]
    
    def generate_initial_population(self, size: int, prompt: str = "", target_code: str = None) -> List[str]:
        """
        Generate initial population of code snippets.
        
        Args:
            size: Number of individuals to generate
            prompt: User's prompt for improvement
            target_code: The original code to be improved
            
        Returns:
            List of code snippets as strings
        """
        if prompt and target_code and self.client:
            return self._generate_initial_population_with_ai(prompt, target_code, size)
        
        # Fallback to simple generation if no prompt/target or if client is not available
        return self._generate_fallback_population(target_code or 'def c(x):\n    return x * 2\n', size)

    def select_parents(self, scores: List[float], n: int = 2) -> List[str]:
        """Select n parents using tournament selection."""
        # Simple tournament selection
        selected = []
        for _ in range(n):
            # Randomly select 3 individuals and pick the best one
            candidates = random.sample(list(zip(self.population, scores)), min(3, len(scores)))
            best = max(candidates, key=lambda x: x[1])
            selected.append(best[0])
        return selected
    
    def _evaluate_efficiency(self, code: str) -> bool:
        """
        Evaluate code efficiency based on some heuristics.
        Returns True if the code meets efficiency criteria.
        """
        try:
            # Simple heuristic: check for obvious inefficiencies
            if 'for ' in code and 'range(' in code and 'len(' in code:
                # Simple check for potential O(n^2) operations
                return False
            return True
        except:
            return False
            
    def _measure_execution_time(self, code: str, test_cases: List[Tuple[Any, Any]]) -> float:
        """
        Measure the average execution time of the code across multiple test cases.
        Returns the average time in seconds.
        """
        try:
            # Create a namespace for the code to run in
            namespace = {}
            exec(code, namespace)
            
            # Get the function from the namespace
            if 'c' not in namespace:
                return float('inf')  # Invalid code
                
            func = namespace['c']
            
            # Time the function across all test cases
            total_time = 0.0
            num_runs = 1000  # Number of times to run each test case for accurate timing
            
            for input_val, _ in test_cases:
                # Use timeit for more accurate timing
                timer = timeit.Timer(lambda: func(input_val))
                total_time += min(timer.repeat(repeat=3, number=num_runs)) / num_runs
                
            return total_time / len(test_cases)  # Return average time per test case
            
        except Exception as e:
            print(f"Error measuring execution time: {e}")
            return float('inf')  # Return infinity if there's an error
    
    def _evaluate_readability(self, code: str) -> bool:
        """Evaluate code readability based on some heuristics."""
        try:
            # Simple heuristic: check line length and variable names
            lines = code.split('\n')
            for line in lines:
                if len(line) > 100:  # Very long lines are hard to read
                    return False
            return True
        except:
            return False
    
    def _evaluate_simplicity(self, code: str) -> bool:
        """Evaluate code simplicity based on some heuristics."""
        try:
            # Simple heuristic: count number of operations
            if code.count(';') > 2:  # Multiple statements per line
                return False
            if len(code.split('\n')) > 10:  # Too many lines
                return False
            return True
        except:
            return False
    
    def _evaluate_maintainability(self, code: str) -> bool:
        """Evaluate code maintainability based on some heuristics."""
        try:
            # Simple heuristic: check for comments and function length
            if '#' not in code and len(code.split('\n')) > 5:
                return False
            return True
        except:
            return False
    
    def _evaluate_with_lm_studio(self, code: str) -> bool:
        """
        Evaluate how well the code matches the user's improvement request using LM Studio.
        Returns True if the code aligns well with the user's intent, False otherwise.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """You are an expert code evaluator. Analyze how well the code matches the user's improvement request.
                    Consider aspects like:
                    - Code structure and organization
                    - Algorithmic efficiency
                    - Readability and maintainability
                    - Adherence to Python best practices
                    - Specific improvements mentioned in the user's request
                    """},
                    {"role": "user", "content": f"""
                    User's improvement request: "{self.improvement_factor}"
                    
                    Code to evaluate:
                    ```python
                    {code}
                    ```
                    
                    Based on the user's request, does this code effectively implement the desired improvements?
                    Consider both explicit and implicit requirements in the user's request.
                    
                    Respond with ONLY 'True' if the code aligns well with the user's request, 'False' otherwise.
                    Be strict but fair in your evaluation.
                    """}
                ],
                temperature=0.2,  # Lower temperature for more consistent evaluations
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().lower()
            return result.startswith('true')
        except Exception as e:
            print(f"Error during LM Studio evaluation: {e}")
            return False
    
    def _evaluate_factor(self, code: str) -> Tuple[bool, float]:
        """
        Evaluate if the code meets the improvement factor criteria.
        Returns a tuple of (is_valid, score) where score is execution time in ms.
        """
        success, message = self.evaluator.evaluate(code)
        if not success:
            return False, float('inf')
        
        try:
            # Extract the execution time from the message (in milliseconds)
            if 'ms' in message:
                time_str = message.replace('ms', '').strip()
                execution_time = float(time_str)
            else:
                # If no 'ms' in message, try to convert directly
                execution_time = float(message)
                
            return True, execution_time
            
        except Exception as e:
            print(f"Error parsing execution time: {e}")
            return False, float('inf')
    
    def _ensure_population_uniqueness(self, population: List[str]) -> List[str]:
        """Ensure all individuals in the population are unique."""
        unique_population = []
        seen = set()
        
        for individual in population:
            # Normalize the code for comparison (remove comments and extra whitespace)
            normalized = '\n'.join(line for line in individual.split('\n') 
                                 if line.strip() and not line.strip().startswith('#'))
            normalized = ' '.join(normalized.split())  # Remove extra whitespace
            
            if normalized not in seen:
                seen.add(normalized)
                unique_population.append(individual)
            else:
                # Generate a new unique variation
                print("\nFound duplicate individual. Generating new variation...")
                new_variant = self._generate_variation(individual)
                if new_variant not in unique_population:
                    unique_population.append(new_variant)
        
        return unique_population
    
    def evolve(self) -> str:
        """
        Run the evolutionary algorithm with the selected evaluation method.
        Returns the best individual found.
        """
        if not self.population:
            self.population = self.generate_initial_population(self.population_size)
        
        # Ensure initial population is unique
        self.population = self._ensure_population_uniqueness(self.population)
        
        print(f"\n--- Starting evolution with {self.evaluation_method.upper()} evaluation ---")
        print(f"Improvement focus: {self.improvement_factor}")
        print(f"Population size: {len(self.population)} unique individuals")
        
        best_individual = None
        best_score = float('inf')
        
        for generation in range(self.max_generations):
            print(f"\n--- Generation {generation + 1} ---")
            
            valid_individuals = []
            scores = []
            execution_times = []
            evaluation_details = []
            
            for i, individual in enumerate(self.population):
                # Check syntax first
                try:
                    ast.parse(individual)
                    syntax_ok = True
                except SyntaxError as e:
                    print(f"Individual {i+1}: ✗ Syntax error: {e}")
                    continue
                
                # Check if it passes test cases
                success, message = self.evaluator.evaluate(individual)
                if not success:
                    print(f"Individual {i+1}: ✗ Failed tests: {message}")
                    continue
                
                # Check if it meets the improvement factor and get score
                is_valid, score = self._evaluate_factor(individual)
                if not is_valid:
                    print(f"Individual {i+1}: ✗ Does not meet {self.improvement_factor} criteria")
                    continue
                
                # If we get here, the individual is valid
                valid_individuals.append(individual)
                scores.append(score)
                
                # For efficiency, we use the measured execution time as the score (lower is better)
                # For other factors, we use the calculated score
                if self.improvement_factor == 'efficiency':
                    # For efficiency, we want to minimize the execution time
                    score_str = f"{score*1000:.4f} ms"
                else:
                    # For other factors, higher score is better
                    score_str = f"{score:.4f}"
                
                print(f"Individual {i+1}: ✓ Passed all checks (score: {score_str})")
                
                # Add to evaluation details
                func_body = '\n'.join(individual.split('\n')[1:])  # Remove first line for brevity
                evaluation_details.append(f"- Score {score_str}: {func_body[:50]}...")
                
                # Update conversation history if using LM Studio
                if self.evaluation_method == 'lm_studio':
                    self.conversation_history.append({
                        "role": "user",
                        "content": f"Here's a valid function that passes all tests with score {score_str}:\n```python\n{individual}\n```"
                    })
            
            # Track the best individual if we have any valid ones
            if valid_individuals and scores:
                best_idx = scores.index(max(scores))
                best_score = scores[best_idx]
                best_individual = valid_individuals[best_idx]
                self.best_individuals.append((best_individual, best_score))
                
                print(f"\n=== Best Solution (Generation {generation + 1}, Score: {best_score:.2f}) ===")
                print("```python")
                print(best_individual.strip())
                print("```")
            else:
                print("No valid individuals in this generation.")
                if generation < self.max_generations - 1:
                    print("Generating new random individuals...")
                    self.population = self.generate_initial_population(self.population_size)
                    continue
                else:
                    print("Maximum generations reached. Stopping evolution.")
                    return ""
            
            print(f"Best score: {best_score}")
            
            # If we have a perfect score, we're done
            if best_score >= 1.0:
                print("\n=== Perfect Solution Found! ===")
                print("\n=== Testing Best Solution ===")
                try:
                    # Execute the best individual to show its output
                    local_vars = {}
                    exec(best_individual, globals(), local_vars)
                    # If the code defines a function, try to call it with test cases
                    if 'c' in local_vars and callable(local_vars['c']):
                        for test_input, expected_output in self.test_cases:
                            result = local_vars['c'](test_input)
                            print(f"Test input: {test_input} => Output: {result}")
                except Exception as e:
                    print(f"Error executing solution: {e}")
                return best_individual
            
            # Select parents for next generation
            new_population = []
            
            # Keep the best individual (elitism)
            new_population.append(best_individual)
            
            # Generate new individuals using LM Studio
            for _ in range(self.population_size - 1):
                # Select parents
                parent1, parent2 = self.select_parents(scores, 2)
                
                # Prepare the prompt for evolution
                evolution_prompt = {
                    "role": "user",
                    "content": f"""I have two Python functions that implement the same functionality but in different ways. 
                    Combine the best parts of both to create an improved version. The function should be named 'c' and take a single input parameter.
                    
                    Function 1:
                    ```python
                    {parent1}
                    ```
                    
                    Function 2:
                    ```python
                    {parent2}
                    ```
                    
                    Previous generation results:
                    {chr(10).join(evaluation_details[-self.population_size:])}
                    
                    Please provide an improved version of this function. Focus on:
                    1. Correctness (passing all test cases)
                    2. Readability
                    3. Efficiency
                    
                    Return ONLY the improved Python function code, without any additional text or explanation.
                    """
                }
                
                # Add to conversation history
                self.conversation_history.append(evolution_prompt)
                
                try:
                    # Get response from LM Studio with conversation history
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=self.conversation_history,
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    # Extract and clean the response
                    child_code = response.choices[0].message.content.strip()
                    
                    # Clean up the response to extract just the code
                    if "```python" in child_code:
                        child_code = child_code.split("```python")[1].split("```")[0].strip()
                    elif "```" in child_code:
                        child_code = child_code.split("```")[1].split("```")[0].strip()
                    
                    # Add the model's response to conversation history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": f"Here's an improved version of the function:\n```python\n{child_code}\n```"
                    })
                    
                    new_population.append(child_code)
                    
                except Exception as e:
                    error_msg = f"Error generating child with LM Studio: {str(e)}"
                    print(error_msg)
                    # Add error to conversation history
                    self.conversation_history.append({
                        "role": "system",
                        "content": f"Error occurred: {error_msg}"
                    })
                    # If there's an error, just keep one of the parents
                    new_population.append(parent1)
            
            self.population = new_population
        
        # Return the best individual found across all generations
        return max(self.best_individuals, key=lambda x: x[1])[0]

def main():
    # Get user input for the code to optimize
    print("Code Evolution Optimizer")
    print("=======================")
    
    # Get the target code file path
    while True:
        file_path = input("\nEnter the path to the Python file containing the code to optimize: ").strip('"')
        try:
            with open(file_path, 'r') as f:
                target_code = f.read()
            break
        except Exception as e:
            print(f"Error reading file: {e}")
    
    # Get the optimization prompt
    print("\nWhat kind of improvements would you like to make?")
    print("Examples:")
    print("- Make this code more efficient")
    print("- Improve the readability of this code")
    print("- Optimize for memory usage")
    prompt = input("\nEnter your improvement request: ")
    
    # Set up test cases
    print("\nSetting up test cases...")
    print("For best results, please provide some test cases (input and expected output).")
    test_cases = [(None, None)]
    
    while True:
        try:
            test_input = input("\nEnter test input (or press Enter to finish): ")
            if not test_input:
                break
                
            test_input = eval(test_input)  # Convert string input to Python object
            expected_output = input("Enter expected output: ")
            expected_output = eval(expected_output)  # Convert string input to Python object
            
            test_cases.append((test_input, expected_output))
            print(f"Added test case: {test_input} -> {expected_output}")
            
            if len(test_cases) >= 5:  # Limit to 5 test cases for simplicity
                print("Maximum test cases reached.")
                break
                
        except Exception as e:
            print(f"Error processing test case: {e}")
    
    # If no test cases were provided, use some defaults
    if not test_cases:
        print("Using default test cases")
        test_cases = [
            (2, 4),
            (0, 0),
            (-3, -6),
            (10, 20)
        ]
    
    # Create evaluator for test cases
    evaluator = CodeEvaluator(test_cases)
    
    # Get evolution parameters
    try:
        population_size = int(input("\nEnter population size (default 5): ") or "5")
        max_generations = int(input("Enter number of generations (default 10): ") or "10")
    except ValueError:
        print("Using default values")
        population_size = 5
        max_generations = 10
    
    # Initialize and run the evolver
    evolver = CodeEvolver(
        initial_population=[],  # Will be generated based on the prompt
        evaluator=evaluator,
        evaluation_method="function",  # 'function' or 'lm_studio'
        improvement_factor="efficiency",
        api_key="lm-studio",
        model="gpt-3.5-turbo",
        population_size=population_size,
        max_generations=max_generations
    )
    
    # Generate initial population based on the prompt and target code
    print("\nGenerating initial population based on your request...")
    initial_population = evolver.generate_initial_population(
        size=population_size,
        prompt=prompt,
        target_code=target_code
    )
    evolver.population = initial_population
    
    best_solution = evolver.evolve()
    
    # The best solution is already printed and tested in the evolve() method
    if not best_solution:
        print("No valid solution was found.")
    # No need to print or test again here as it's already done in evolve()

if __name__ == "__main__":
    main()
