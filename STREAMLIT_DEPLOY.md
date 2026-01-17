# Streamlit Deployment Guide

## 🚀 Deploy to Streamlit Community Cloud (FREE)

### Prerequisites
1. Push your repo to GitHub (use `./push_to_github.sh`)
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Sign in with your GitHub account

### Deployment Steps

1. **Click "New app"**

2. **Configure deployment:**
   - **Repository**: `Mounusha25/automl-tabular`
   - **Branch**: `main`
   - **Main file path**: `app/streamlit_app.py`
   - **App URL**: Choose a custom URL (e.g., `automl-tabular-demo`)

3. **Click "Deploy"**

   It will take 2-3 minutes to install dependencies and start the app.

4. **You're live!** 🎉
   
   Your app will be at: `https://[your-app-name].streamlit.app`

### Testing Locally First

Before deploying, test locally:

```bash
# Install Streamlit (if not already)
pip install streamlit

# Run the app
streamlit run app/streamlit_app.py
```

This will open `http://localhost:8501` in your browser.

### Update Deployment

After pushing changes to GitHub:
- Streamlit Cloud automatically detects changes
- Click "Reboot app" in the Streamlit Cloud dashboard to apply updates

---

## 🎨 Customization

### Update Theme Colors

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#your-color"
backgroundColor = "#your-color"
```

### Update App Title

Edit `app/streamlit_app.py`, line 17:

```python
st.set_page_config(
    page_title="Your Title",
    page_icon="🤖",
)
```

---

## 📊 Usage Limits (Free Tier)

- **Resources**: 1 GB RAM, shared CPU
- **Apps**: Unlimited public apps
- **Users**: Unlimited viewers
- **Uptime**: Sleeps after inactivity, wakes on first request

**Tip**: Set `time_budget` lower (30-60 seconds) in demo mode to avoid timeouts.

---

## 🔗 Add to Your Resume/README

Once deployed, add this to your README:

```markdown
## 🌐 Live Demo

Try the interactive demo: **[AutoML Tabular Demo](https://your-app.streamlit.app)**

Upload your own CSV or try example datasets (Titanic, Wine Quality, California Housing).
```

And on your resume:

```
AutoML Tabular | Live Demo: your-app.streamlit.app
Production AutoML engine with interactive web interface for exploring tabular ML tasks
```

---

## 🐛 Troubleshooting

**App crashes on deployment:**
- Check logs in Streamlit Cloud dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify file paths are correct (use `Path(__file__).parent`)

**App is slow:**
- Reduce `max_trials_per_model` in config (already set to 5 for demo)
- Lower `time_budget` slider max value
- Add `@st.cache_data` decorator for data loading

**Import errors:**
- Make sure `src/` directory structure is preserved
- Check `sys.path.insert()` in `streamlit_app.py` line 13

---

## 🎯 Next Steps

1. Deploy to Streamlit Cloud
2. Test with example datasets
3. Share the link with recruiters!
4. Consider adding:
   - Example predictions section
   - Model comparison charts
   - Download report as HTML
