# Admin Setup Instructions - Shared Workspace

## CRITICAL: Shared Workspace Access Method

**DO NOT use presigned URLs for the shared workspace.** Presigned URLs create separate user-specific JupyterLab instances, defeating the entire cost optimization goal.

**ONLY METHOD:** AWS Console navigation (below)

---

## Initial Setup Steps

### Step 1: Access the Shared Workspace (Admin)

1. Login to AWS Console:
   ```
   https://535146832369.signin.aws.amazon.com/console
   ```
   - Use your admin AWS credentials (not student credentials)

2. Navigate to SageMaker:
   - Search for "SageMaker" in the console
   - Click **Amazon SageMaker**
   - Go to **Domains** → `bread-financial-academy`
   - Click **Spaces** tab
   - Click on `shared-academy-workspace`
   - Click **"Open JupyterLab"**

3. Wait 2-3 minutes for the JupyterLab app to launch (cold start)

### Step 2: Create Student Folders

Once JupyterLab is open:

1. Open a **Terminal** in JupyterLab (File → New → Terminal)

2. Upload the folder setup script:
   - Download `scripts/setup_student_folders.py` from the repo
   - Use JupyterLab's Upload button to upload it

3. Run the script:
   ```bash
   python3 setup_student_folders.py
   ```

4. Verify output:
   ```
   Creating admin folder + 66 student folders in: /home/sagemaker-user
   --------------------------------------------------------
   ✓ Created admin-axel/ with README.md
   ✓ Created student1/ with README.md
   ✓ Created student2/ with README.md
   ...
   ✓ Created student66/ with README.md
   --------------------------------------------------------

   ✅ Folder setup complete!
      - Created: 67 folders
      - Skipped: 0 folders (already existed)

   Folders created:
      - Admin: admin-axel/
      - Students: student1/, student2/, ..., student66/

   🎓 All 67 users (admin + 66 students) can now use their assigned folders!
   ```

5. Verify folders appear in the JupyterLab file browser

---

## Student Access Distribution

### Credentials CSV

Located at: `infrastructure/sagemaker/terraform/persistent/student-credentials.csv`

Contains:
- Cohort assignment
- Full name
- Username (student1 - student66)
- Password (auto-generated)
- Login URL

**Send this file to students BEFORE the first Friday class.**

### Access Instructions

Send students the guide: `infrastructure/sagemaker/STUDENT_ACCESS_GUIDE.md`

**Key points to emphasize:**
1. ✅ Login via AWS Console (URL in credentials CSV)
2. ✅ Navigate: SageMaker → Domains → Spaces → shared-academy-workspace → Open JupyterLab
3. ❌ DO NOT use presigned URLs (they create separate instances)
4. ✅ Use file naming convention: `studentX_weekY_description.ipynb`
5. ✅ Save work to their assigned folder: `studentX/`

---

## Friday Class Workflow

### Before Class (10:00 Buenos Aires Time)

1. **Admin:** Access the shared workspace via console
   - This ensures the JupyterLab app is already running
   - Students will have instant access (no 2-3 minute wait)

2. **Admin:** Verify student folders exist
   - Check that all 66 student folders + admin folder are present
   - If not, run `setup_student_folders.py` again

3. **Students:** Login and navigate via console
   - First few students may wait 1-2 minutes if app isn't running
   - Once running, all subsequent students get instant access

### During Class (10:00-20:00)

- **All 67 users** (admin + 66 students) access the SAME JupyterLab instance
- Students work in their assigned folders: `student1/`, `student2/`, etc.
- Admin uses `admin-axel/` for demos and teaching materials
- Everyone can see each other's files (collaborative environment)

### After Class (20:00+)

- Students close their browsers
- JupyterLab app continues running at **$0.10/hour** (ml.m5.large)

### Saturday 00:00 Buenos Aires Time

- **Lambda cleanup function** automatically deletes the JupyterLab app
- **EFS files preserved** (all student notebooks remain intact)
- Cost stops until next Friday class

---

## Cost Optimization Verification

### Expected Costs

**Single shared instance approach:**
- Friday 10:00-20:00: 10 hours × $0.10/hour = **$1.00/week**
- Total for 12-week program: **$12.00**

**What we AVOIDED (67 separate instances):**
- 67 instances × 10 hours × $0.10/hour = **$67.00/week**
- Total for 12-week program: **$804.00**

**Savings: $792.00 (98% reduction)**

### How to Verify

1. After Friday class, check running apps:
   ```bash
   aws sagemaker list-apps \
     --domain-id d-cakhetabszon \
     --space-name shared-academy-workspace \
     --profile di-mfa
   ```

2. Should see **ONLY ONE** JupyterLab app:
   ```json
   {
       "Apps": [
           {
               "AppType": "JupyterLab",
               "AppName": "default",
               "Status": "InService",
               "CreationTime": "2026-01-10T13:00:00Z"
           }
       ]
   }
   ```

3. If you see multiple apps with different user profiles → **PROBLEM**
   - Students used presigned URLs instead of console navigation
   - Delete extra apps manually
   - Re-emphasize console navigation to students

---

## Troubleshooting

### Problem: Student doesn't see folders

**Cause:** Student used presigned URL instead of console navigation

**Fix:**
1. Delete the user-specific JupyterLab app they created
2. Have student login via console and navigate to shared workspace
3. They should now see all 67 folders

### Problem: Multiple JupyterLab apps running

**Cause:** Students used presigned URLs

**Fix:**
1. List all apps:
   ```bash
   aws sagemaker list-apps \
     --domain-id d-cakhetabszon \
     --space-name shared-academy-workspace \
     --profile di-mfa
   ```

2. Delete user-specific apps (keep only the one everyone should use):
   ```bash
   aws sagemaker delete-app \
     --domain-id d-cakhetabszon \
     --space-name shared-academy-workspace \
     --app-type JupyterLab \
     --app-name default \
     --profile di-mfa
   ```

3. Have everyone re-access via console navigation

### Problem: Lambda cleanup didn't run

**Check:**
```bash
aws logs tail /aws/lambda/sagemaker-studio-app-cleanup \
  --since 1d \
  --profile di-mfa
```

**Expected output (Saturday 00:00):**
```
Deleted app: shared-academy-workspace/JupyterLab/default
Cleanup complete: 1 apps deleted
```

---

## Week 5-7 Notebook Distribution

### Notebooks Created

- ✅ Week 5: AI Services (Comprehend, Textract, Rekognition)
- ✅ Week 6: Call Center ML (Transcribe, Comprehend, XGBoost)
- ✅ Week 7: MLflow Monitoring (Model tracking, monitoring)

### Distribution Method

**Download link:**
```
https://courses.axel.net.s3.amazonaws.com/Bread%20Financial%20Academy/Week%205-6-7%20Sagemaker/week_5-6-7.zip
```

**Student instructions:**
1. Download the zip file
2. Extract notebooks
3. Login to AWS Console → Navigate to shared workspace
4. Upload notebooks to their assigned folder (e.g., `student15/`)
5. Run notebooks during Friday class

---

## Questions?

Contact: Axel Sirota (admin-axel@bread-financial-academy.com)
