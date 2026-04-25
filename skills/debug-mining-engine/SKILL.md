# Debug Mining Engine

**"The answer lies in the darkness"** - Transform every debugging session into reusable skills

## Overview

The Debug Mining Engine automatically captures debugging sessions and transforms them into three reusable assets:
1. **Full Skills** - Comprehensive methodologies with scripts and templates
2. **Code Snippets** - Quick copy/paste solutions
3. **Pattern Guides** - Conceptual understanding documents

Every bug you fix becomes a permanent asset in your skill arsenal.

## When to Use This Skill

Use the Debug Mining Engine when you want to:
- **Preserve debugging knowledge** instead of losing it
- **Build reusable solutions** from errors you've fixed
- **Create a growing library** of debugging patterns
- **Reduce time** spent on similar issues in the future
- **Transform failures into assets**

## The Philosophy

Traditional debugging:
- Fix the bug → Move on → Forget the solution
- Repeat the same debugging process later
- Knowledge lost, time wasted

Debug Mining:
- Fix the bug → Capture the session → Generate skills
- Reuse the solution next time
- Knowledge preserved, time saved

**ROI**: Every debugging session becomes an investment that pays dividends forever.

## How It Works

### Hybrid Detection System

**AI Auto-Detection** (Passive):
- Monitors shell for error patterns
- Detects failures automatically
- Starts background capture
- Prompts you after detecting a fix

**Manual Override** (Active):
```bash
/debug-start "Description of what you're debugging"
# ... debug and fix the issue ...
/debug-end
```

**Smart Prompts**:
```
🤔 I noticed you just fixed an error. Should I save this debugging session?
   Error: "Could not find table 'repositories'"
   Solution: Created schema validation script
   
   [Yes, save it] [No, discard] [Let me review first]
```

### What Gets Captured

- **Error Context**: Full error message, stack trace, exit code
- **Solution Journey**: All attempts (including failures)
- **Final Solution**: The code/command that worked
- **Environment**: Working directory, relevant files
- **Metadata**: Duration, timestamps, tags

### What Gets Generated

**1. Full Skill** (`/home/ubuntu/skills/skills/debugging-patterns/[skill-name]/`)
```
├── SKILL.md                    # Complete methodology
├── scripts/
│   ├── detect_error.py         # Error detection
│   ├── apply_fix.py            # Automated fix
│   └── validate_solution.py    # Validation
├── templates/
│   ├── fix_template.sh         # Template for similar fixes
│   └── test_template.py        # Test template
└── references/
    ├── error_analysis.md       # Deep dive
    ├── solution_rationale.md   # Why this works
    └── related_patterns.md     # Similar issues
```

**2. Code Snippet** (`/home/ubuntu/debug-snippets/[category]/[name].sh`)
```bash
#!/bin/bash
# Quick Fix: Supabase Schema Validation
# Error: Could not find table 'repositories'
# Solution: Validate schema before querying

python3 << 'EOF'
from supabase import create_client
client = create_client(url, key)

# Validate table exists
try:
    client.table('your_table').select('*').limit(1).execute()
    print("✅ Table exists")
except Exception as e:
    print(f"❌ Table not found: {e}")
EOF
```

**3. Pattern Guide** (`/home/ubuntu/debug-patterns/[name].md`)
```markdown
# Pattern: Supabase Schema Validation

## The Problem
Querying Supabase tables without validating schema first

## Why This Happens
- Table names change during development
- Schema migrations not applied

## The Solution Pattern
Always validate schema before querying

## When to Apply
- Before any Supabase query
- After schema migrations
```

## Setup

### Step 1: Install the Monitoring System

```bash
# Run the setup script
bash /home/ubuntu/skills/skills/utility-skills/debug-mining-engine/scripts/setup_monitor.sh
```

This adds debug monitoring to your shell environment.

### Step 2: Create Storage Directories

```bash
mkdir -p /home/ubuntu/debug-sessions
mkdir -p /home/ubuntu/debug-snippets/{database,api,filesystem,network,auth}
mkdir -p /home/ubuntu/debug-patterns
```

### Step 3: Test the System

```bash
# Trigger a test error
/debug-start "Testing debug mining"
python3 -c "raise Exception('Test error')"
/debug-end

# Check if session was captured
ls /home/ubuntu/debug-sessions/$(date +%Y-%m-%d)/
```

## Usage Examples

### Example 1: Automatic Capture

```bash
# You run a command that fails
$ python3 audit_script.py
❌ Error: Could not find table 'repositories'

# AI detects error and starts capturing
[Debug Mining: Capture started]

# You try different solutions
$ python3 audit_script.py --table users
❌ Still failing

$ python3 audit_script.py --validate-schema
✅ Success!

# AI detects fix and prompts
🤔 Should I save this debugging session?
[Yes, save it]

# AI generates assets
✅ Generated 3 assets:
   1. Full Skill: supabase-schema-validator
   2. Code Snippet: supabase-schema-check.sh
   3. Pattern Guide: database-validation.md
```

### Example 2: Manual Capture

```bash
# Start debugging manually
$ /debug-start "Fixing API connection timeout"
[Debug Mining: Manual capture started]

# Debug and fix
$ curl https://api.example.com/endpoint
Error: Connection timeout

$ curl --retry 3 --retry-delay 2 https://api.example.com/endpoint
✅ Success!

# End capture
$ /debug-end
[Debug Mining: Analyzing session...]

✅ Generated skill: api-retry-with-backoff
```

### Example 3: Using Generated Skills

```bash
# List available debugging skills
ls /home/ubuntu/skills/skills/debugging-patterns/

# Use a code snippet
bash /home/ubuntu/debug-snippets/database/supabase-schema-check.sh

# Read a pattern guide
cat /home/ubuntu/debug-patterns/database-validation.md
```

## Intelligence Features

### Pattern Recognition

The system learns from your debugging history:

```
🔍 Pattern Detected!
   You've debugged "table not found" errors 3 times this month.
   
   Suggestion: Create a pre-flight validation skill
   
   [Create skill] [Remind me later]
```

### Proactive Suggestions

```
💡 Based on your debugging history:
   - "api-connection-validator" (you debug API errors often)
   - "environment-config-checker" (env vars cause 40% of errors)
   
   [Generate these skills] [Show examples]
```

### Learning Metrics

```
📈 Debug Mining Stats (Last 30 Days)
   
   Debugging Time: 12 hours → 3 hours (-75%)
   Errors Captured: 45
   Skills Generated: 12
   Reuse Count: 28
   Time Saved: ~9 hours
```

## Scripts Reference

### Core Scripts

**`setup_monitor.sh`** - Install shell monitoring system
```bash
bash scripts/setup_monitor.sh
```

**`monitor.py`** - Error detection and session capture
```bash
python3 scripts/monitor.py --exit-code 1 --command "failed_command"
```

**`analyzer.py`** - Pattern analysis and extraction
```bash
python3 scripts/analyzer.py --session-id debug_2026-02-11_01-33-08
```

**`generator.py`** - Multi-format skill generation
```bash
python3 scripts/generator.py --session-id debug_2026-02-11_01-33-08
```

### Utility Scripts

**`list_sessions.py`** - List captured debugging sessions
```bash
python3 scripts/list_sessions.py --days 30
```

**`stats.py`** - Show debug mining statistics
```bash
python3 scripts/stats.py
```

**`search_patterns.py`** - Search for similar patterns
```bash
python3 scripts/search_patterns.py --error "table not found"
```

## Templates Reference

**`skill_template/`** - Template for generated full skills
**`snippet_template.sh`** - Template for code snippets
**`pattern_template.md`** - Template for pattern guides

## Best Practices

### 1. Add Context When Starting Manual Capture

```bash
# Good
/debug-start "Fixing Supabase query error in audit script"

# Not as good
/debug-start "debugging"
```

### 2. Review Generated Skills

Always review generated skills before adding to arsenal:
```bash
# Review before committing
cat /home/ubuntu/skills/skills/debugging-patterns/new-skill/SKILL.md
```

### 3. Tag Your Sessions

Add tags to make skills discoverable:
```bash
/debug-start "Fixing API timeout" --tags api,network,timeout
```

### 4. Update Existing Skills

If you find a better solution, update the skill:
```bash
python3 scripts/update_skill.py --skill supabase-schema-validator --session debug_2026-02-12_10-30-00
```

### 5. Share Patterns

Export patterns to share with team:
```bash
python3 scripts/export_pattern.py --pattern database-validation --format markdown
```

## Troubleshooting

### Monitor Not Detecting Errors

**Problem**: Errors not being captured automatically

**Solution**:
1. Check if monitor is installed: `which debug_monitor_command`
2. Reload shell: `source ~/.bashrc`
3. Test manually: `/debug-start "test"`

### Sessions Not Saving

**Problem**: Debug sessions not appearing in `/home/ubuntu/debug-sessions/`

**Solution**:
1. Check directory permissions: `ls -la /home/ubuntu/debug-sessions/`
2. Check monitor logs: `tail -f /home/ubuntu/debug-mining/monitor.log`
3. Run monitor manually: `python3 scripts/monitor.py --test`

### Generated Skills Have Errors

**Problem**: Generated skills contain incorrect code

**Solution**:
1. Review the captured session: `cat /home/ubuntu/debug-sessions/[date]/[session-id].json`
2. Regenerate with corrections: `python3 scripts/generator.py --session [id] --review`
3. Edit manually and mark as reviewed

## Success Metrics

### Capture Quality
- **Accuracy**: 95%+ of debugging sessions correctly detected
- **Completeness**: 90%+ of sessions with full context
- **False Positives**: <5% non-debugging sessions captured

### Skill Quality
- **Reusability**: Generated skills used 3+ times on average
- **Time Saved**: 2-5 hours saved per reused skill
- **Pattern Coverage**: 80%+ of errors have matching patterns

### System Impact
- **Debugging Time Reduction**: 50-75% decrease
- **Knowledge Retention**: 100% of debugging knowledge preserved
- **Arsenal Growth**: 10-20 new skills per month

## Advanced Features

### Custom Analyzers

Create custom pattern analyzers for your specific domain:

```python
# /home/ubuntu/debug-mining/custom_analyzers/my_analyzer.py

from analyzer import PatternAnalyzer

class MyCustomAnalyzer(PatternAnalyzer):
    def analyze_custom_pattern(self, session):
        # Your custom analysis logic
        pass
```

### Integration with Skill Arsenal

Generated skills automatically integrate with your unified skill arsenal:

```python
from lib.skill_registry import SkillRegistry

registry = SkillRegistry('/home/ubuntu/skills/skills.json')

# Find debugging skills
debug_skills = registry.find_by_category('debugging-patterns')

# Get related skills
related = registry.get_complements('supabase-schema-validator')
```

### Workflow Automation

Chain debugging skills with other skills:

```python
from lib.skill_composer import SkillComposer

composer = SkillComposer(registry)

# Create workflow
workflow = composer.compose_workflow([
    'brainstorming',
    'writing-plans',
    'database-schema-generator',
    'supabase-schema-validator',  # Generated from debugging!
    'testing-framework'
])

composer.execute_workflow(workflow, context)
```

## Future Enhancements

- **AI-powered root cause analysis** using LLMs
- **Cross-project pattern recognition**
- **Team collaboration** (share debugging patterns)
- **Visual debugging timeline**
- **Integration with systematic-debugging skill**

## Related Skills

- **systematic-debugging** - Methodical debugging approach
- **skill-arsenal-builder** - Build unified skill collections
- **skill-creator** - Create new skills from scratch
- **feature-verification** - Verify fixes work correctly

## Time Investment

- **Setup**: 15 minutes (one-time)
- **Per debugging session**: 0 minutes (automatic)
- **Reviewing generated skills**: 5-10 minutes
- **ROI**: 50-75% reduction in debugging time

## Key Takeaway

**Every bug you fix becomes a permanent asset.**

Stop losing debugging knowledge. Start building a library of solutions that grows with every error you encounter.

**"The answer lies in the darkness"** - And now you can capture it! 🎯
