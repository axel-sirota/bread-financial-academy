# SageMaker Studio Shared Workspace - Student Access Guide

## Overview

All 66 students share **ONE** JupyterLab workspace for cost optimization. When you access the workspace, you'll join the same JupyterLab instance as your classmates.

**Cost savings:** 1 shared instance vs 67 separate instances = 98% cost reduction!

---

## Access Instructions

### Step 1: Login to AWS Console

1. Open your browser and navigate to:
   ```
   https://535146832369.signin.aws.amazon.com/console
   ```

2. Enter your credentials:
   - **IAM username:** `studentX` (where X is your student number, e.g., `student15`)
   - **Password:** (from the `student-credentials.csv` file provided)

**CRITICAL:** You MUST use the AWS Console navigation path above. Do NOT use presigned URLs - they create separate instances and defeat the cost optimization goal.

### Step 2: Navigate to SageMaker Studio

1. Once logged in, search for "SageMaker" in the AWS Console search bar
2. Click on **Amazon SageMaker**
3. In the left sidebar, under "Admin configurations", click **Domains**
4. Click on the domain: `bread-financial-academy`
5. Click on the **"Spaces"** tab at the top

### Step 3: Open Shared Workspace

1. You'll see one space listed: `shared-academy-workspace`
2. Click on the space name
3. Click the **"Open JupyterLab"** button

**⏳ Wait Time:**
- **First student:** 2-3 minutes (cold start - launching the instance)
- **Subsequent students:** INSTANT ACCESS (app already running)

### Step 4: Upload and Run Your Notebook

1. Once JupyterLab opens, you'll see a file browser on the left
2. Click the **Upload** button (up arrow icon) to upload your `.ipynb` notebook
3. Navigate to where you saved your notebook (e.g., `week5_ai_services_exploration.ipynb`)
4. Click to open the notebook and start working!

---

## File Naming Convention

**IMPORTANT:** To avoid confusion with 66 students in the same workspace, use this naming convention:

```
studentX_weekY_description.ipynb
```

**Examples:**
- `student15_week5_ai_services.ipynb`
- `student15_week6_call_center.ipynb`
- `student42_week7_deployment.ipynb`

---

## Working in the Shared Environment

### What's Shared?
- ✅ **JupyterLab Application:** All students use the same running instance
- ✅ **File Storage (EFS):** You can see each other's files
- ✅ **Compute Resources:** One ml.m5.large instance serves everyone

### What's NOT Shared?
- ❌ **Jupyter Kernels:** Each student runs their own Python kernel
- ❌ **Notebook Execution:** Your code runs independently
- ❌ **S3 Data:** Each student saves their results separately

### Can I See Other Students' Notebooks?
Yes! This is intentional. You can:
- View classmates' notebooks for learning
- Help each other debug
- Share example code

**⚠️ Please respect others' work:** Don't modify or delete files that aren't yours!

---

## Collaboration Features

SageMaker Studio supports **real-time collaboration:**
- Multiple students can open the SAME notebook simultaneously
- Each user gets a colored cursor with their username
- Changes are visible in real-time (like Google Docs)

This is great for:
- Pair programming during class
- Reviewing each other's work
- Debugging together

---

## Friday Class Schedule (10:00-20:00 Buenos Aires Time)

**10:00 - First Student Arrives:**
- Clicks "Open JupyterLab" → Waits 2-3 minutes for cold start
- Instance launches → Gets to work

**10:15+ - Other Students Arrive:**
- Click "Open JupyterLab" → INSTANT ACCESS
- Instance already running from first student

**20:00 - Class Ends:**
- Close your browser
- JupyterLab keeps running (no action needed)

**Saturday 00:00 - Automatic Cleanup:**
- Lambda function deletes the JupyterLab app
- Your files remain safe on EFS storage
- Cost stops ($0/hour until next Friday)

**Next Friday 10:00 - Cycle Repeats:**
- First student starts the app again
- Everyone else gets instant access

---

## Cost Efficiency

| Scenario | Cost per 10-hour session |
|----------|-------------------------|
| 67 separate instances | 67 × $0.05 × 10 = **$33.50** |
| 1 shared instance | 1 × $0.05 × 10 = **$0.50** |
| **Savings** | **$33.00 per week!** |

Over 12 weeks: **$396 saved!**

---

## Troubleshooting

### Issue: "Access Denied" when opening workspace
**Solution:** Contact your instructor. You may need IAM permissions updated.

### Issue: JupyterLab taking a long time to launch
**Solution:** First student after Saturday cleanup experiences cold start (2-3 min). This is normal!

### Issue: Can't find my notebook
**Solution:** Check if you're in the shared workspace. Your notebooks are in the EFS file browser (left sidebar).

### Issue: Another student deleted my work
**Solution:** This shouldn't happen! Contact your instructor immediately. All students should follow naming conventions and respect others' files.

---

## Best Practices

1. **Use Clear Naming:** Always prefix with your student number
2. **Save Often:** Your work auto-saves, but manually save important checkpoints
3. **Close Unused Kernels:** Help conserve resources for classmates
4. **Be Respectful:** Don't modify or delete others' files
5. **Collaborate Wisely:** Use real-time collaboration for pair work during class

---

## Getting Help

**During Class:**
- Ask your instructor for help
- Collaborate with classmates in the same workspace

**Technical Issues:**
- Contact your instructor
- Check AWS CloudWatch logs (if you have permission)

---

## Summary

✅ **Login:** AWS Console → SageMaker → Domains → Spaces → shared-academy-workspace
✅ **Launch:** Click "Open JupyterLab"
✅ **Upload:** Your `studentX_weekY.ipynb` notebooks
✅ **Work:** Run code, access AWS AI Services, save to S3
✅ **Collaborate:** Real-time co-editing with classmates
✅ **Save Money:** 1 instance for 67 students = 98% cost reduction!

Happy Learning! 🚀
