---
name: systematic-feature-builder
description: Systematic approach to implementing features with detailed planning, tracking, and testing. Use when building new features, adding functionality to existing projects, implementing multi-step requirements, or when user requests organized, methodical development workflow.
---

# Systematic Feature Builder

Implement features systematically with detailed planning, progress tracking, and comprehensive testing.

## When to Use This Skill

Use this skill when:
- Building new features from requirements or specifications
- Adding functionality to existing projects
- Implementing multi-step features that require coordination
- User requests "systematic", "organized", or "methodical" approach
- User provides a spec or detailed requirements document
- Features require database changes, API integration, or complex logic

## Core Workflow

Follow these five phases in order:

1. **Requirements Analysis** - Understand what needs to be built
2. **Implementation Planning** - Create detailed, step-by-step plan
3. **Systematic Implementation** - Execute plan with progress tracking
4. **Testing & Verification** - Validate at multiple levels
5. **Documentation & Checkpoint** - Create comprehensive checkpoint

## Phase 1: Requirements Analysis

### Gather Information

Extract clear requirements from user input:
- What features need to be built?
- What is the expected behavior?
- Are there existing examples or references?
- What are the success criteria?

### Identify Dependencies

Determine what exists and what needs to be created:
- Existing code/components that can be reused
- Database schema changes required
- External services or APIs needed
- Dependencies between features

### Clarify Ambiguities

Ask targeted questions if requirements are unclear:
- Edge cases and error handling
- User permissions and access control
- Performance or scalability requirements
- UI/UX preferences

## Phase 2: Implementation Planning

### Create Detailed Plan

Use the implementation plan template:

```bash
# Copy template to project
cp /home/ubuntu/skills/systematic-feature-builder/templates/implementation-plan-template.md \
   <project-path>/docs/plans/YYYY-MM-DD-feature-name.md
```

**Plan Structure:**
- Overview and objectives
- Task breakdown with estimated time
- Step-by-step instructions with file paths
- Testing strategy for each task
- Success criteria
- Rollback plan

### Break Down Into Atomic Tasks

Each task should:
- Be completable in one sitting (30-90 minutes)
- Have clear inputs and outputs
- Be independently testable
- Have explicit success criteria

**Example Task Breakdown:**

```markdown
### Task 1: Add Database Field (15 min)
1. Update schema.ts with new field
2. Run `pnpm db:push` to migrate
3. Verify field exists in database

### Task 2: Create Backend Endpoint (30 min)
1. Add endpoint to router
2. Implement business logic
3. Add input validation
4. Write unit tests

### Task 3: Build Frontend Component (45 min)
1. Create component file
2. Add tRPC integration
3. Implement UI with shadcn/ui
4. Handle loading/error states
```

### Update todo.md

Add all planned tasks to project's todo.md:

```markdown
## [Feature Name]

- [ ] Task 1: Database changes
- [ ] Task 2: Backend endpoint
- [ ] Task 3: Frontend component
- [ ] Task 4: Testing
- [ ] Task 5: Documentation
```

## Phase 3: Systematic Implementation

### Execute Plan Sequentially

Implement tasks in order, following the plan:

1. **Start with backend/data layer** (database, APIs)
2. **Then business logic** (endpoints, services)
3. **Finally UI layer** (components, pages)

This order minimizes rework and allows incremental testing.

### Track Progress in todo.md

Mark tasks complete as you finish them:

```markdown
## [Feature Name]

- [x] Task 1: Database changes
- [x] Task 2: Backend endpoint
- [ ] Task 3: Frontend component  ← Currently working on this
- [ ] Task 4: Testing
- [ ] Task 5: Documentation
```

Use `file` tool's `edit` action to update checkboxes efficiently.

### Handle Deviations

If you deviate from the plan:
- Document why in the plan file
- Update todo.md with new tasks
- Adjust time estimates if needed

### Communicate Progress

For long implementations, send `info` messages at milestones:
- "✅ Task 1 complete: Database schema updated"
- "📊 Progress: 3/5 tasks complete"
- "⚠️ Encountered issue with X, implementing alternative approach"

## Phase 4: Testing & Verification

### Three-Level Testing

Test at all three levels before checkpoint:

**1. Unit Tests**
- Individual functions and endpoints
- Database operations
- Input validation

**2. Integration Tests**
- Features working together
- End-to-end workflows
- External service integration

**3. Manual Verification**
- UI renders correctly
- User flows work as expected
- Edge cases handled gracefully

### Webdev-Specific Testing

For webdev projects:

```bash
# Run automated tests
pnpm test

# Check dev server health
# (use webdev_check_status tool)

# Manual browser testing
# - Open dev server URL
# - Test feature in browser
# - Check console for errors
# - Verify responsive design
```

Refer to `references/testing-strategies.md` for detailed testing approaches.

### Verify Completion

Before creating checkpoint, verify all tasks complete:

```bash
python /home/ubuntu/skills/systematic-feature-builder/scripts/verify_completion.py \
  <project-path>/todo.md
```

This script checks for incomplete `[ ]` items and reports them.

## Phase 5: Documentation & Checkpoint

### Update Documentation

Ensure documentation reflects changes:
- Update README if user-facing features added
- Add inline code comments for complex logic
- Update API documentation if endpoints changed

### Create Comprehensive Checkpoint

Follow checkpoint guidelines in `references/checkpoint-guidelines.md`.

**Checkpoint Description Template:**

```markdown
[Feature name]: [brief summary]

## Features Added

### 1. [Feature Name]
- Key capability 1
- Key capability 2
- Location/integration point

## Technical Details

**Database Changes:**
- [Schema changes]

**New Files:**
- [Important new files and their purpose]

**Dependencies:**
- [New packages added]

## Testing
- [X] tests passing
- [Y] tests failing (if any, explain why)

## Next Steps
- [Suggested follow-up actions]
```

### Webdev Checkpoint Checklist

Before creating webdev checkpoint:

- [ ] All tests passing (or failures documented)
- [ ] `webdev_check_status` shows no critical errors
- [ ] Feature works in dev server
- [ ] todo.md updated with completed items marked `[x]`
- [ ] Large media files uploaded to S3 (not in project directory)
- [ ] Database migrations applied successfully

## Webdev-Specific Guidance

### Database Changes

Always follow this sequence:

1. Edit `drizzle/schema.ts`
2. Run `pnpm db:push` to apply migration
3. Verify migration succeeded
4. Test database operations

### File Organization

Create files in this order:

1. **Database layer**: `drizzle/schema.ts`
2. **Data access**: `server/db.ts` (query helpers)
3. **Business logic**: `server/*-router.ts` (tRPC endpoints)
4. **Frontend**: `client/src/components/*.tsx`, `client/src/pages/*.tsx`

### Testing Pattern

```typescript
// server/feature.test.ts
import { describe, it, expect } from 'vitest';

describe('Feature Tests', () => {
  it('should handle valid input', async () => {
    // Arrange
    const input = { /* test data */ };
    
    // Act
    const result = await endpoint(input);
    
    // Assert
    expect(result).toMatchObject({ expected });
  });
  
  it('should handle error cases', async () => {
    // Test error scenarios
  });
});
```

### Common Patterns

**Adding a Feature:**
1. Plan → Update todo.md
2. Schema → Database migration
3. Backend → tRPC endpoint + tests
4. Frontend → Component + integration
5. Test → Manual verification
6. Checkpoint → Comprehensive description

**Fixing a Bug:**
1. Reproduce → Document steps
2. Identify → Root cause analysis
3. Fix → Minimal change
4. Test → Verify fix + no regressions
5. Checkpoint → Explain bug and fix

## Best Practices

### Do

✅ **Write plans before coding** - Saves time, reduces rework  
✅ **Update todo.md actively** - Provides clear progress tracking  
✅ **Test incrementally** - Catch issues early  
✅ **Document decisions** - Future you will thank present you  
✅ **Create detailed checkpoints** - Enable easy rollback and review

### Don't

❌ **Skip planning** - Leads to disorganized implementation  
❌ **Implement everything at once** - Hard to debug, test, and review  
❌ **Forget to update todo.md** - Loses track of progress  
❌ **Skip testing** - Bugs slip into production  
❌ **Create vague checkpoints** - Useless for rollback or review

## Templates and Scripts

### Available Resources

**Templates** (`templates/`):
- `implementation-plan-template.md` - Structured feature plan
- `todo-template.md` - Format for tracking features

**Scripts** (`scripts/`):
- `verify_completion.py` - Validates all checklist items complete

**References** (`references/`):
- `testing-strategies.md` - Testing approaches by project type
- `checkpoint-guidelines.md` - Checkpoint best practices

### Using Templates

Copy templates to your project:

```bash
# Implementation plan
cp /home/ubuntu/skills/systematic-feature-builder/templates/implementation-plan-template.md \
   <project>/docs/plans/$(date +%Y-%m-%d)-feature-name.md

# Todo file (if project doesn't have one)
cp /home/ubuntu/skills/systematic-feature-builder/templates/todo-template.md \
   <project>/todo.md
```

## Examples

### Example: Adding Bulk CSV Import Feature

**Phase 1: Requirements**
- Admin needs to import rep codes from CSV
- CSV format: email, repCode
- Show import results with errors

**Phase 2: Plan**
```markdown
### Task 1: Backend Endpoint (30 min)
1. Add bulkImportRepCodes to admin-router.ts
2. Parse CSV data from input
3. Validate each row
4. Update database
5. Return results with errors

### Task 2: Frontend Component (45 min)
1. Create BulkRepCodeImport.tsx
2. Add file upload input
3. Integrate Papa Parse
4. Show import results
5. Add CSV template download

### Task 3: Integration (15 min)
1. Add component to Admin page
2. Test with sample CSV
3. Verify error handling
```

**Phase 3: Implementation**
- Implement tasks sequentially
- Update todo.md after each task
- Test each component independently

**Phase 4: Testing**
- Unit test: endpoint validates CSV correctly
- Integration test: upload CSV via UI
- Manual test: download template, upload, verify results

**Phase 5: Checkpoint**
```
Title: Add bulk rep code CSV import

## Features Added
- Admin endpoint for bulk importing rep codes
- BulkRepCodeImport component with file upload
- Papa Parse integration
- Detailed error reporting

## Technical Details
**New Files:**
- server/admin-router.ts: bulkImportRepCodes endpoint
- client/src/components/BulkRepCodeImport.tsx

**Dependencies:**
- papaparse, @types/papaparse

## Testing
- 52 tests passing
```

## Troubleshooting

**"Plan is too detailed/taking too long"**
→ Reduce granularity. Focus on major steps, not every line of code.

**"Implementation deviating from plan"**
→ Update plan document with deviations and reasoning. Adjust todo.md.

**"Tests failing after implementation"**
→ Fix tests before checkpoint. Document any intentional test changes.

**"Checkpoint description too long"**
→ Focus on what changed and why. Move detailed technical notes to plan document.

## Summary

This skill provides a systematic, organized approach to feature implementation:

1. **Analyze** requirements thoroughly
2. **Plan** implementation with detailed steps
3. **Implement** systematically with progress tracking
4. **Test** at multiple levels
5. **Document** comprehensively in checkpoint

The result: high-quality, well-tested features with clear documentation and easy rollback capability.
