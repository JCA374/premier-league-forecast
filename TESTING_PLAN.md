# Streamlit Cloud Deployment Testing Plan

## Goal
Find which part of the app is causing the Streamlit Cloud deployment failure by testing incrementally.

## Backup
- Original app: `app.py.backup`
- Restore command: `cp app.py.backup app.py`

## Test Sequence

### Test 1: Absolute Minimum ✓
**File**: `app_test1_minimal.py`
- Just "Hello World"
- Verifies: Streamlit Cloud can run ANY app at all

### Test 2: Basic Imports ✓
**File**: `app_test2_imports.py`
- Add pandas, numpy, plotly
- Verifies: Standard scientific packages work

### Test 3: Custom Module Imports ✓
**File**: `app_test3_modules.py`
- Import shl.data.scraper
- Import shl.data.cleaner
- Import shl.models.poisson_model
- etc.
- Verifies: Our custom modules can be imported

### Test 4: Database Init ✓
**File**: `app_test4_database.py`
- Add DatabaseManager initialization
- Verifies: Database setup works in cloud environment

### Test 5: Full Session State ✓
**File**: `app_test5_session.py`
- Add all session state initialization
- Verifies: Session state setup doesn't crash

### Test 6: Navigation Structure ✓
**File**: `app_test6_navigation.py`
- Add sidebar navigation
- Empty page functions (just st.write)
- Verifies: Navigation framework works

### Test 7+: Add Pages One by One ✓
- Test 7: Data Collection page
- Test 8: Data Verification page
- Test 9: Model Training page
- Test 10: Odds Integration page
- Test 11: Monte Carlo Simulation page
- Test 12: Fixture Predictions page
- Test 13: Results Analysis page
- Test 14: Dashboard page
- Test 15: Database Management page

## Testing Process

For each test:
1. Create test file
2. Update main app.py with test code
3. Commit and push
4. Check Streamlit Cloud deployment
5. If SUCCESS: proceed to next test
6. If FAIL: identified the problem! Debug that specific part

## Restore Command
```bash
cp app.py.backup app.py
git add app.py
git commit -m "Restore original app.py"
git push origin claude/init-project-011CUoX6Tg8BGHL3hchRrgBR
```
