# Performance Comparison: Claude Code with Deepseek/Gemini vs Anthropic

This document provides a comprehensive performance comparison between using the actual Anthropic API versus our proxy implementation with Deepseek and Gemini models.

## Summary of Findings

Our benchmark tests show the following performance characteristics:

1. **For simple text generation (Haiku → Gemini Flash):**
   - Gemini Flash is actually **faster** than Claude Haiku API
   - Average response time: ~0.7s (vs ~1.0s for Anthropic)
   - Performance ratio: Proxy is 0.63-0.80x the response time of Anthropic
   - Net difference: ~0.2-0.4s faster than Anthropic

2. **For complex reasoning (Sonnet → Deepseek):**
   - Deepseek is significantly slower than Anthropic's Opus model
   - Average response time: ~15-17s
   - Anthropic's Opus took ~7-8s for similar tasks

3. **For code generation (Sonnet → Deepseek):**
   - Deepseek is slower than Anthropic's Opus model
   - Average response time: ~15-20s
   - Anthropic's Opus took ~16s for similar tasks

## Performance Formula

Based on the test results, we can use the following formula to estimate task completion times for the proxy versus direct Claude:

For Haiku/simple tasks (mapped to Gemini Flash):
- **T_proxy ≈ 0.7 × T_anthropic**

For Sonnet/complex tasks (mapped to Deepseek):
- **T_proxy ≈ 2.0-2.5 × T_anthropic**

Where:
- T_proxy = Expected time using the proxy
- T_anthropic = Expected time using direct Anthropic API

## Detailed Benchmark Results

| Test Case | Anthropic API | Proxy (Deepseek/Gemini) | Ratio | Difference |
|-----------|--------------|-------------------------|-------|------------|
| Simple text (Haiku) | 0.99-1.50s | 0.67-0.79s | 0.63-0.80x | -0.2 to -0.4s |
| Complex reasoning (Sonnet) | ~7.8s (Opus) | 15-17s | ~2.0x | +7-9s |
| Code generation (Sonnet) | ~16.2s (Opus) | 15-20s | ~1.0-1.2x | +0-4s |

## Impact on Coding Tasks

For a typical coding session, the impact will vary based on the types of interactions:

1. **Simple queries, navigation, and tool usage**:
   - Will be approximately the same speed or slightly faster with the proxy
   - Example: "Show me file X", "What does function Y do?", "List all files in directory Z"

2. **Complex reasoning and deep analysis**:
   - Will be approximately 2-2.5x slower with the proxy
   - Example: "Why is this code causing a race condition?", "How does this algorithm work?"

3. **Code generation**:
   - Will be approximately 1-1.2x slower with the proxy
   - Example: "Generate a function to parse JSON", "Write a test for this class"

## Server Configuration Impact

Running with `--always-cot` flag showed:
- No significant impact on Gemini Flash (simple tasks) performance
- Improved quality for Deepseek (complex tasks), especially for reasoning tasks

## Overall Assessment

The proxy provides an excellent alternative to Claude with minimal performance impact for simple and quick interactions, which typically constitute the majority of coding assistive tasks. The most noticeable slowdown occurs during complex reasoning tasks, but even these remain within acceptable limits for practical usage.

For most coding tasks, the performance tradeoff is well balanced against the potential cost savings of using alternative models.