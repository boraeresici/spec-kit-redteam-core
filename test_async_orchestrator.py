#!/usr/bin/env python3
"""
Load Testing for AsyncAgentOrchestrator

Comprehensive load testing suite to validate concurrent execution,
performance under stress, and system scalability.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from spec_kit_redteam_plugin.async_orchestrator import AsyncCollaborativeOrchestrator
from spec_kit_redteam_plugin.token_tracker import TokenTracker
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()


@dataclass
class LoadTestResult:
    """Load test execution results"""
    test_name: str
    concurrent_requests: int
    total_requests: int
    success_count: int
    failure_count: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    throughput_rps: float
    total_duration: float
    memory_usage_mb: float
    error_rate: float
    performance_summary: Dict[str, Any]


class AsyncOrchestratorLoadTester:
    """Load testing suite for AsyncAgentOrchestrator"""
    
    def __init__(self):
        self.console = console
        self.test_results: List[LoadTestResult] = []
    
    async def run_concurrent_generation_test(
        self, 
        concurrent_requests: int = 5,
        requests_per_concurrent: int = 2
    ) -> LoadTestResult:
        """Test concurrent specification generations"""
        
        test_name = f"Concurrent Generation ({concurrent_requests}x{requests_per_concurrent})"
        console.print(f"\n🧪 [bold cyan]Starting {test_name}[/bold cyan]")
        
        # Test configurations
        test_descriptions = [
            "web application with user authentication and payment processing",
            "microservice architecture for e-commerce platform",
            "mobile app with real-time chat and notifications", 
            "API gateway with rate limiting and monitoring",
            "distributed system with message queues and caching"
        ]
        
        agents_configs = [
            ['pm', 'technical', 'security'],
            ['pm', 'security', 'qa'],
            ['technical', 'security', 'devops'],
            ['pm', 'technical', 'qa', 'security'],
            ['technical', 'security', 'compliance']
        ]
        
        # Performance metrics
        response_times = []
        success_count = 0
        failure_count = 0
        start_time = time.time()
        
        # Create concurrent tasks
        async def single_generation_request(request_id: int) -> Dict[str, Any]:
            try:
                request_start = time.time()
                
                # Vary the test parameters
                description = test_descriptions[request_id % len(test_descriptions)]
                agents = agents_configs[request_id % len(agents_configs)]
                
                # Create orchestrator with monitoring
                token_tracker = TokenTracker(initial_budget=50.0)
                
                async with AsyncCollaborativeOrchestrator(
                    token_tracker=token_tracker,
                    enable_caching=True,
                    enable_parallel=True,
                    max_concurrent_agents=4,
                    connection_timeout=30.0,
                    console=Console()
                ) as orchestrator:
                    
                    result = await orchestrator.generate_collaborative_spec(
                        description=description,
                        agents=agents,
                        max_budget=25.0,
                        complexity='medium',
                        enable_streaming=True
                    )
                    
                    request_time = time.time() - request_start
                    
                    return {
                        'request_id': request_id,
                        'success': True,
                        'response_time': request_time,
                        'agents_used': len(agents),
                        'spec_length': len(result.specification) if result.specification else 0,
                        'tokens_used': result.token_summary.get('total_tokens', 0),
                        'cost': result.token_summary.get('total_cost', 0.0),
                        'quality_score': result.quality_score
                    }
                    
            except Exception as e:
                request_time = time.time() - request_start
                console.print(f"❌ Request {request_id} failed: {str(e)[:100]}")
                
                return {
                    'request_id': request_id,
                    'success': False,
                    'response_time': request_time,
                    'error': str(e),
                    'agents_used': 0,
                    'spec_length': 0,
                    'tokens_used': 0,
                    'cost': 0.0,
                    'quality_score': 0.0
                }
        
        # Execute concurrent batches
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Running {concurrent_requests} concurrent batches...", 
                total=concurrent_requests
            )
            
            batch_tasks = []
            for batch_id in range(concurrent_requests):
                # Each batch runs multiple requests
                batch_requests = [
                    single_generation_request(batch_id * requests_per_concurrent + i)
                    for i in range(requests_per_concurrent)
                ]
                batch_tasks.append(asyncio.gather(*batch_requests))
            
            # Execute all batches concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for batch_idx, batch_result in enumerate(batch_results):
                progress.update(task, advance=1)
                
                if isinstance(batch_result, Exception):
                    console.print(f"❌ Batch {batch_idx} failed: {batch_result}")
                    failure_count += requests_per_concurrent
                else:
                    for request_result in batch_result:
                        if request_result['success']:
                            success_count += 1
                            response_times.append(request_result['response_time'])
                        else:
                            failure_count += 1
        
        # Calculate metrics
        total_duration = time.time() - start_time
        total_requests = concurrent_requests * requests_per_concurrent
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else avg_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0.0
        
        throughput_rps = success_count / total_duration if total_duration > 0 else 0
        error_rate = (failure_count / total_requests) * 100 if total_requests > 0 else 0
        
        result = LoadTestResult(
            test_name=test_name,
            concurrent_requests=concurrent_requests,
            total_requests=total_requests,
            success_count=success_count,
            failure_count=failure_count,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            throughput_rps=throughput_rps,
            total_duration=total_duration,
            memory_usage_mb=0.0,  # Could add psutil for memory monitoring
            error_rate=error_rate,
            performance_summary={}
        )
        
        self.test_results.append(result)
        self._display_test_result(result)
        
        return result
    
    async def run_stress_test(self, max_concurrent: int = 10) -> LoadTestResult:
        """Run stress test with increasing load"""
        
        test_name = f"Stress Test (up to {max_concurrent} concurrent)"
        console.print(f"\n💪 [bold red]Starting {test_name}[/bold red]")
        
        stress_results = []
        
        # Gradually increase concurrent load
        for concurrent_level in range(2, max_concurrent + 1, 2):
            console.print(f"\n📈 Testing with {concurrent_level} concurrent requests...")
            
            result = await self.run_concurrent_generation_test(
                concurrent_requests=concurrent_level,
                requests_per_concurrent=1
            )
            
            stress_results.append({
                'concurrent_level': concurrent_level,
                'throughput': result.throughput_rps,
                'avg_response_time': result.avg_response_time,
                'error_rate': result.error_rate
            })
            
            # Stop if error rate gets too high
            if result.error_rate > 50:
                console.print(f"⚠️  Stopping stress test - error rate too high: {result.error_rate:.1f}%")
                break
            
            # Brief pause between stress levels
            await asyncio.sleep(2)
        
        # Find optimal concurrency level
        optimal_level = max(
            stress_results,
            key=lambda x: x['throughput'] if x['error_rate'] < 10 else 0
        )
        
        console.print(f"\n🎯 [bold green]Optimal concurrency level: {optimal_level['concurrent_level']} "
                     f"(Throughput: {optimal_level['throughput']:.2f} RPS, "
                     f"Error Rate: {optimal_level['error_rate']:.1f}%)[/bold green]")
        
        # Return summary as LoadTestResult
        return LoadTestResult(
            test_name=test_name,
            concurrent_requests=max_concurrent,
            total_requests=sum(len(stress_results) for _ in stress_results),
            success_count=sum(1 for r in stress_results if r['error_rate'] < 10),
            failure_count=sum(1 for r in stress_results if r['error_rate'] >= 10),
            avg_response_time=statistics.mean([r['avg_response_time'] for r in stress_results]),
            min_response_time=min([r['avg_response_time'] for r in stress_results]),
            max_response_time=max([r['avg_response_time'] for r in stress_results]),
            p95_response_time=0.0,
            throughput_rps=optimal_level['throughput'],
            total_duration=0.0,
            memory_usage_mb=0.0,
            error_rate=optimal_level['error_rate'],
            performance_summary={'optimal_concurrent_level': optimal_level['concurrent_level']}
        )
    
    async def run_endurance_test(self, duration_minutes: int = 5) -> LoadTestResult:
        """Run endurance test for sustained load"""
        
        test_name = f"Endurance Test ({duration_minutes} minutes)"
        console.print(f"\n⏰ [bold yellow]Starting {test_name}[/bold yellow]")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        response_times = []
        success_count = 0
        failure_count = 0
        
        request_id = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task(
                f"Running endurance test for {duration_minutes} minutes...",
                total=100
            )
            
            while time.time() < end_time:
                try:
                    # Run a small batch every few seconds
                    batch_start = time.time()
                    
                    async with AsyncCollaborativeOrchestrator(
                        enable_caching=True,
                        enable_parallel=True,
                        max_concurrent_agents=3
                    ) as orchestrator:
                        
                        result = await orchestrator.generate_collaborative_spec(
                            description=f"Test request {request_id} - web application with basic features",
                            agents=['pm', 'technical', 'security'],
                            max_budget=10.0,
                            complexity='low'
                        )
                        
                        response_time = time.time() - batch_start
                        response_times.append(response_time)
                        success_count += 1
                    
                except Exception as e:
                    failure_count += 1
                    console.print(f"❌ Endurance request {request_id} failed: {str(e)[:50]}")
                
                request_id += 1
                
                # Update progress
                elapsed_pct = ((time.time() - start_time) / (duration_minutes * 60)) * 100
                progress.update(task, completed=min(elapsed_pct, 100))
                
                # Brief pause between requests
                await asyncio.sleep(5)
        
        total_duration = time.time() - start_time
        
        # Calculate metrics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
        
        total_requests = success_count + failure_count
        throughput_rps = success_count / total_duration if total_duration > 0 else 0
        error_rate = (failure_count / total_requests) * 100 if total_requests > 0 else 0
        
        result = LoadTestResult(
            test_name=test_name,
            concurrent_requests=1,
            total_requests=total_requests,
            success_count=success_count,
            failure_count=failure_count,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=0.0,
            throughput_rps=throughput_rps,
            total_duration=total_duration,
            memory_usage_mb=0.0,
            error_rate=error_rate,
            performance_summary={'duration_minutes': duration_minutes}
        )
        
        self.test_results.append(result)
        self._display_test_result(result)
        
        return result
    
    def _display_test_result(self, result: LoadTestResult):
        """Display formatted test result"""
        
        # Create results table
        table = Table(title=f"📊 {result.test_name} Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Requests", str(result.total_requests))
        table.add_row("Successful", f"{result.success_count} ({100 - result.error_rate:.1f}%)")
        table.add_row("Failed", f"{result.failure_count} ({result.error_rate:.1f}%)")
        table.add_row("Avg Response Time", f"{result.avg_response_time:.2f}s")
        table.add_row("Min Response Time", f"{result.min_response_time:.2f}s")
        table.add_row("Max Response Time", f"{result.max_response_time:.2f}s")
        table.add_row("Throughput", f"{result.throughput_rps:.2f} RPS")
        table.add_row("Total Duration", f"{result.total_duration:.2f}s")
        
        console.print(table)
        
        # Performance assessment
        if result.error_rate < 5 and result.avg_response_time < 60:
            status = "[bold green]✅ EXCELLENT[/bold green]"
        elif result.error_rate < 15 and result.avg_response_time < 120:
            status = "[bold yellow]⚠️  ACCEPTABLE[/bold yellow]"
        else:
            status = "[bold red]❌ NEEDS IMPROVEMENT[/bold red]"
        
        console.print(f"\n📈 Performance Assessment: {status}")
    
    def display_summary(self):
        """Display comprehensive test summary"""
        
        if not self.test_results:
            console.print("[yellow]No test results to display[/yellow]")
            return
        
        console.print("\n" + "="*80)
        console.print("🏁 [bold cyan]LOAD TESTING SUMMARY[/bold cyan]")
        console.print("="*80)
        
        # Summary table
        summary_table = Table(title="All Test Results")
        summary_table.add_column("Test Name")
        summary_table.add_column("Requests")
        summary_table.add_column("Success Rate")
        summary_table.add_column("Avg Response")
        summary_table.add_column("Throughput")
        summary_table.add_column("Status")
        
        for result in self.test_results:
            success_rate = f"{100 - result.error_rate:.1f}%"
            avg_response = f"{result.avg_response_time:.2f}s"
            throughput = f"{result.throughput_rps:.2f} RPS"
            
            if result.error_rate < 5:
                status = "✅"
            elif result.error_rate < 15:
                status = "⚠️"
            else:
                status = "❌"
            
            summary_table.add_row(
                result.test_name,
                str(result.total_requests),
                success_rate,
                avg_response,
                throughput,
                status
            )
        
        console.print(summary_table)
        
        # Overall assessment
        avg_error_rate = sum(r.error_rate for r in self.test_results) / len(self.test_results)
        avg_throughput = sum(r.throughput_rps for r in self.test_results) / len(self.test_results)
        
        console.print(f"\n📊 [bold]Overall Performance:[/bold]")
        console.print(f"   • Average Error Rate: {avg_error_rate:.1f}%")
        console.print(f"   • Average Throughput: {avg_throughput:.2f} RPS")
        
        if avg_error_rate < 5:
            console.print(f"   • 🎉 [bold green]System performs excellently under load![/bold green]")
        elif avg_error_rate < 15:
            console.print(f"   • 👍 [bold yellow]System handles load reasonably well[/bold yellow]")
        else:
            console.print(f"   • 🔧 [bold red]System needs optimization for better load handling[/bold red]")


async def main():
    """Run comprehensive load testing suite"""
    
    console.print(Panel.fit(
        "[bold cyan]🚀 AsyncAgentOrchestrator Load Testing Suite[/bold cyan]\n"
        "Testing concurrent execution, performance, and scalability",
        border_style="blue"
    ))
    
    tester = AsyncOrchestratorLoadTester()
    
    try:
        # Test 1: Basic concurrent generation
        await tester.run_concurrent_generation_test(
            concurrent_requests=3,
            requests_per_concurrent=2
        )
        
        # Test 2: Higher concurrency  
        await tester.run_concurrent_generation_test(
            concurrent_requests=5,
            requests_per_concurrent=2
        )
        
        # Test 3: Stress test (if not in CI/limited environment)
        console.print("\n🤔 [yellow]Running stress test (this may take a few minutes)...[/yellow]")
        await tester.run_stress_test(max_concurrent=8)
        
        # Test 4: Short endurance test
        console.print("\n🕒 [yellow]Running short endurance test...[/yellow]")
        await tester.run_endurance_test(duration_minutes=2)
        
    except KeyboardInterrupt:
        console.print("\n⚠️  [yellow]Load testing interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n❌ [red]Load testing failed: {e}[/red]")
    finally:
        # Always show summary
        tester.display_summary()


if __name__ == "__main__":
    asyncio.run(main())