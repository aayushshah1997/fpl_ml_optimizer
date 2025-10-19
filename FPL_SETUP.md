# FPL Team Data Setup Guide

## 🎯 **The Datetime Issue is Fixed!**

Great news! The "can't subtract offset-naive and offset-aware datetimes" error has been completely resolved. Your dashboard should now load team data successfully once you set up the required credentials.

## 📋 **What You Need to Do**

To load your FPL team data, you need to set up authentication credentials. Here's how:

### **Step 1: Create Environment Variables**

Create a `.env` file in your project root directory with the following content:

```bash
# FPL (Fantasy Premier League) Credentials
# Required for accessing your private FPL team data

FPL_EMAIL=your_fpl_email@example.com
FPL_PASSWORD=your_fpl_password
FPL_ENTRY_ID=your_team_id_number

# FBRef API Key (optional, for enhanced stats)
FBR_API_KEY=your_fbr_api_key
```

### **Step 2: Get Your FPL Credentials**

1. **FPL_EMAIL**: Your Fantasy Premier League login email
2. **FPL_PASSWORD**: Your Fantasy Premier League password
3. **FPL_ENTRY_ID**: Your team ID number (found in the URL when viewing your team on the FPL website)

### **Step 3: Get Your FPL Team ID**

Your team ID can be found in the URL when you visit your team on the official FPL website:
- Go to https://fantasy.premierleague.com/
- Log in to your account
- Navigate to your team
- Look at the URL: `https://fantasy.premierleague.com/entry/YOUR_TEAM_ID/event/1`
- The number after `/entry/` is your team ID

### **Step 4: Test the Setup**

Once you've created the `.env` file with your credentials, restart your dashboard and try loading your team data again.

## 🔒 **Security Notes**

- Your `.env` file contains sensitive information - never commit it to version control
- The credentials are only used to authenticate with the official FPL API
- All data is cached locally for performance
- No credentials are stored permanently or transmitted to third parties

## 🚀 **What You'll Get**

Once set up, the dashboard will display:
- ✅ Your current team squad with all players
- ✅ Player prices and price changes
- ✅ Current gameweek information and deadlines
- ✅ Upcoming fixtures with difficulty ratings
- ✅ Team strength analysis and metrics
- ✅ Bank balance and team value

## 🆘 **Troubleshooting**

If you still get authentication errors:
1. Double-check your email and password
2. Verify your team ID is correct
3. Ensure the `.env` file is in the project root directory
4. Restart the dashboard completely

The datetime timezone errors are now completely resolved! 🎉






