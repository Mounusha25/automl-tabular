# 🎯 Streamlit Demo Guide

**Quick demo for recruiters and stakeholders**  
**Zero changes to core logic - pure UX layer**

---

## 🚀 Quick Start

```bash
# Install Streamlit (if not already installed)
pip3 install streamlit --user

# Run demo
streamlit run demo_streamlit.py
```

The demo will open in your browser at `http://localhost:8501`

---

## 🎥 30-Second Demo Flow

**For recruiters - the "wow" factor:**

1. **Landing Page** (5 seconds)
   - See clean, professional UI
   - Performance metrics: 1.37x speedup, 100% quality
   - Clear value proposition

2. **Select Dataset** (5 seconds)
   - Click "Titanic (Binary Classification)"
   - See data preview (891 rows, 12 columns)

3. **Run AutoML** (15 seconds)
   - Click "🚀 Run AutoML Pipeline"
   - Watch progress bar with live updates
   - See automatic problem detection

4. **View Results** (5 seconds)
   - Best model: logistic_regression
   - Score: 0.8850 ROC_AUC
   - Model leaderboard with all trials
   - Download report/model

**Total**: 30 seconds from zero to trained model ✨

---

## ✨ What Recruiters See

### Visual Impact
- ✅ Clean, modern gradient UI
- ✅ Real-time progress tracking
- ✅ Professional metrics display
- ✅ Embedded HTML report preview
- ✅ One-click downloads

### Technical Depth
- ✅ Automatic task detection (Binary/Multiclass/Regression)
- ✅ Optuna hyperparameter optimization
- ✅ Performance metrics (1.37x speedup)
- ✅ Model leaderboard with trials
- ✅ Explainable AI (feature importance, selection rationale)

### Production Quality
- ✅ Error handling
- ✅ Progress feedback
- ✅ Clean code structure
- ✅ Documentation
- ✅ Optimized performance

---

## 📊 Features

### Main Tab: "🚀 Run Demo"

**Step 1: Choose Dataset**
- Pre-loaded: Titanic (Binary Classification)
- Or upload custom CSV
- Automatic data validation

**Step 2: Data Preview**
- Row/column count metrics
- Target column display
- Expandable data sample table

**Step 3: Run AutoML**
- Configurable trials (5/10/20)
- Live progress bar with status
- Automatic problem type detection
- Real-time updates

**Step 4: Results**
- Best model card
- Score + search time metrics
- Full model leaderboard
- Download HTML report
- Download trained model (.joblib)
- Embedded report preview

### About Tab: "📊 About"

- Key features overview
- Technical stack
- Performance benchmarks
- Documentation links
- Project status

---

## 🎯 Use Cases

### For Interviews
**Talking Points:**
- "Built production-grade AutoML from scratch"
- "Optimized for 1.37x speedup with zero quality loss"
- "Explainable AI with HTML reports"
- "Smart preprocessing with data-driven strategies"
- "Clean architecture: orchestrator, search, preprocessing, reporting"

### For Portfolio
- Embed in GitHub README (screenshots + live demo link)
- Deploy to Streamlit Cloud (free hosting)
- Show in LinkedIn project showcase
- Include in resume projects section

### For Demonstrations
- Show ML Engineering best practices
- Highlight performance optimization
- Demonstrate explainability focus
- Showcase clean UX design

---

## 🚀 Deployment (Optional)

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo
4. Set main file: `demo_streamlit.py`
5. Deploy (takes ~2 minutes)
6. Get public URL: `https://your-app.streamlit.app`

### Local Sharing

```bash
# Run on specific port
streamlit run demo_streamlit.py --server.port 8080

# Share on local network
streamlit run demo_streamlit.py --server.address 0.0.0.0
```

---

## 📝 Customization

### Add More Datasets

Edit [demo_streamlit.py](demo_streamlit.py):

```python
dataset_option = st.radio(
    "Select a sample dataset or upload your own:",
    [
        "Titanic (Binary Classification)",
        "Adult Income (Binary Classification)",  # Add this
        "California Housing (Regression)",       # Add this
        "Upload Custom CSV"
    ],
    horizontal=True
)
```

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Adjust Trials

Default is 5 trials per model (fast demo). Change in UI or code:

```python
n_trials = st.selectbox("Trials per model:", [5, 10, 20], index=0)
```

---

## 🎬 Recording Demo (Optional)

### For Video Portfolio

1. **Open browser to localhost:8501**
2. **Start screen recording** (QuickTime on Mac, OBS on Windows)
3. **Narrate while clicking:**
   - "This is AutoML Tabular, production-grade AutoML"
   - "Select Titanic dataset... 891 rows detected"
   - "Run AutoML... watch automatic task detection"
   - "Completed in 7 seconds, found best model"
   - "Here's the full HTML report with explanations"
4. **Export as MP4** (30-60 seconds max)
5. **Upload to LinkedIn/YouTube**

### For Screenshots

1. Run demo
2. Capture key screens:
   - Landing page
   - Data preview
   - Progress bar
   - Results dashboard
   - Report preview
3. Use in README, resume, LinkedIn

---

## ⚠️ Notes

### Core Logic
- **Zero changes** to AutoML core (`src/automl_tabular/`)
- Pure presentation layer
- Safe to run alongside frozen v1.1.0

### Performance
- First run may be slower (model compilation)
- Subsequent runs faster (cache)
- Progress bar updates in real-time

### Datasets
- Titanic path: `examples/titanic.csv`
- If missing, upload CSV manually
- Custom datasets fully supported

---

## 🎯 What This Demonstrates

**To recruiters/hiring managers:**

1. **ML Engineering** (not just data science)
   - Performance optimization (1.37x speedup)
   - Production architecture
   - Clean abstractions

2. **Full-Stack ML**
   - Backend: AutoML engine
   - Frontend: Streamlit UI
   - DevOps: Documentation, testing, versioning

3. **Explainability Focus**
   - HTML reports with rationale
   - Feature importance
   - Model selection transparency

4. **Professional Quality**
   - Error handling
   - Progress feedback
   - Clean UX
   - Comprehensive docs

---

**Perfect for: Interviews, portfolio, LinkedIn showcase, live demos** ✨
