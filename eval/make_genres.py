"""Generate a corpus of spreadsheets across the genres people actually upload.

Architectural schedules, bills of quantity, P&L statements, budgets, shopping lists,
inventories, sales pivots, todo lists, payroll registers, directories, lookup tables,
category hierarchies, timesheets, gradebooks, expense claims and mixed workbooks.

Each carries the mess its genre really has: merged group headers, subtotal rows,
title blocks, indented hierarchies, checkbox columns, months as columns, and a form
header sitting above a table.

    python eval/make_genres.py && python eval/score_genres.py
"""
import pathlib

from openpyxl import Workbook

D = pathlib.Path("eval/fixtures/genres")
D.mkdir(parents=True, exist_ok=True)

def xl(name, sheets):
    """Write one workbook, one worksheet per entry."""
    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title[:31])
        for row in rows:
            ws.append(row)
    wb.save(D / name)

def csv(name: str, text: str) -> None:
    (D / name).write_text(text)

# 1. architectural room schedule -- merged two-row header, units row
xl("arch_room_schedule.xlsx", {"Rooms": [
    ["PROJECT: Tower B — Level 3 Room Schedule"], [],
    ["Room", "Room", "Finishes", "Finishes", "Finishes", "Area"],
    ["No.", "Name", "Floor", "Wall", "Ceiling", "sqm"],
    ["301", "Lobby", "Granite", "Paint", "Gypsum", 42.5],
    ["302", "Office A", "Vinyl", "Paint", "Grid", 28.0],
    ["303", "Office B", "Vinyl", "Paint", "Grid", 28.0],
    ["304", "Toilet", "Ceramic", "Tile", "Gypsum", 9.25],
    ["305", "Server", "Raised", "Paint", "Grid", 15.0],
    [], ["", "", "", "", "TOTAL", 122.75],
]})

# 2. bill of quantities -- hierarchical item numbers, section subtotals
xl("boq.xlsx", {"BOQ": [
    ["BILL OF QUANTITIES"], ["Contract:", "CN-2291", "", "Rev", "C"], [],
    ["Item", "Description", "Unit", "Qty", "Rate", "Amount"],
    ["1", "EARTHWORKS", "", "", "", ""],
    ["1.1", "Site clearance", "sqm", 1200, 45, 54000],
    ["1.2", "Excavation to reduce level", "cum", 340, 280, 95200],
    ["", "", "", "", "Subtotal", 149200],
    ["2", "CONCRETE", "", "", "", ""],
    ["2.1", "Blinding 75mm", "sqm", 410, 390, 159900],
    ["2.2", "RCC grade M30", "cum", 96, 6200, 595200],
    ["", "", "", "", "Subtotal", 755100],
]})

# 3. P&L -- months across columns, category rows
xl("pnl.xlsx", {"P&L": [
    ["Profit & Loss — FY2026"], [],
    ["", "Apr", "May", "Jun", "Jul", "Aug", "Sep"],
    ["Revenue", 120000, 135000, 128000, 142000, 151000, 149000],
    ["Cost of sales", 62000, 68000, 65000, 71000, 74000, 73000],
    ["Gross profit", 58000, 67000, 63000, 71000, 77000, 76000],
    ["Salaries", 31000, 31000, 32000, 32000, 33000, 33000],
    ["Rent", 8000, 8000, 8000, 8000, 8000, 8000],
    ["Net profit", 19000, 28000, 23000, 31000, 36000, 35000],
]})

# 4. budget tracker -- category groups with indentation and subtotals
csv("budget_tracker.csv",
    "Category,Item,Budget,Actual,Variance\n"
    "Housing,,,,\n"
    ",Rent,25000,25000,0\n"
    ",Electricity,3000,3450,-450\n"
    ",Internet,1200,1200,0\n"
    "Food,,,,\n"
    ",Groceries,12000,13800,-1800\n"
    ",Dining out,4000,2600,1400\n"
    "Transport,,,,\n"
    ",Fuel,6000,5400,600\n"
    ",Maintenance,2000,0,2000\n")

# 5. shopping list -- tiny, informal
csv("shopping_list.csv",
    "Item,Qty,Where,Done\n"
    "Milk 1L,2,Local,yes\n"
    "Eggs,12,Local,no\n"
    "Coffee beans,1,Market,no\n"
    "Rice 5kg,1,Wholesale,yes\n"
    "Detergent,2,Local,no\n")

# 6. inventory -- SKU table plus a small summary block below
xl("inventory.xlsx", {"Stock": [
    ["SKU", "Product", "Category", "On hand", "Reorder at", "Unit cost"],
    ["SKU-1001", "Blue widget", "Widgets", 142, 50, 12.5],
    ["SKU-1002", "Red widget", "Widgets", 18, 50, 12.5],
    ["SKU-2001", "Steel bracket", "Fittings", 640, 200, 3.75],
    ["SKU-2002", "Brass bracket", "Fittings", 94, 200, 6.1],
    ["SKU-3001", "Cable 2m", "Cables", 0, 100, 2.2],
    [], [],
    ["Summary"], ["Total SKUs", 5], ["Below reorder", 3],
]})

# 7. sales pivot -- wide, months as columns, region rows
csv("sales_pivot.csv",
    "Region,Jan,Feb,Mar,Apr,May,Jun,Total\n"
    "North,41000,38000,45000,47000,44000,49000,264000\n"
    "South,52000,55000,51000,58000,61000,59000,336000\n"
    "East,33000,31000,36000,38000,35000,40000,213000\n"
    "West,47000,49000,46000,51000,53000,50000,296000\n")

# 8. todo list -- checkboxes, priorities, blanks
csv("todo.csv",
    "Done,Task,Owner,Due,Priority\n"
    "x,Draft proposal,Asha,2026-03-04,High\n"
    ",Review budget,Ravi,2026-03-08,High\n"
    ",Book venue,Lin,2026-03-12,Medium\n"
    "x,Send invites,Asha,2026-02-28,Low\n"
    ",Order catering,,2026-03-15,Medium\n")

# 9. payroll register -- multi-row header, employee rows, totals row
xl("payroll_register.xlsx", {"March": [
    ["PAYROLL REGISTER — March 2026"], [],
    ["Emp", "Name", "Earnings", "Earnings", "Earnings", "Deductions", "Deductions", "Net"],
    ["ID", "", "Basic", "HRA", "Allow", "PF", "Tax", "Pay"],
    ["E001", "Asha Menon", 45000, 18000, 6000, 5400, 4200, 59400],
    ["E002", "Ravi Kumar", 38000, 15200, 5000, 4560, 3100, 50540],
    ["E003", "Lin Wei", 52000, 20800, 7000, 6240, 5800, 67760],
    ["", "TOTAL", 135000, 54000, 18000, 16200, 13100, 177700],
]})

# 10. directory -- contacts, some blanks
csv("directory.csv",
    "Name,Department,Email,Phone,Location\n"
    "Asha Menon,Engineering,asha@example.com,+91 98400 11111,Chennai\n"
    "Ravi Kumar,Sales,ravi@example.com,,Hyderabad\n"
    "Lin Wei,Engineering,lin@example.com,+91 98400 33333,Bengaluru\n"
    "Omar Haddad,Finance,,+91 98400 44444,Mumbai\n")

# 11. lookup table -- two columns, code to description
csv("lookup_codes.csv",
    "Code,Description\n"
    "AC,Air conditioning\n"
    "EL,Electrical\n"
    "PL,Plumbing\n"
    "FF,Fire fighting\n"
    "CV,Civil works\n"
    "HV,HVAC ducting\n")

# 12. category divisions -- indented hierarchy in one column
csv("categories.csv",
    "Category,Count\n"
    "Electronics,240\n"
    "  Phones,120\n"
    "  Laptops,80\n"
    "  Accessories,40\n"
    "Apparel,180\n"
    "  Mens,90\n"
    "  Womens,70\n"
    "  Kids,20\n")

# 13. timesheet -- dates across columns
xl("timesheet.xlsx", {"Week 12": [
    ["Timesheet — week commencing 2026-03-16"], [],
    ["Employee", "Mon", "Tue", "Wed", "Thu", "Fri", "Total"],
    ["Asha Menon", 8, 8, 7.5, 8, 6, 37.5],
    ["Ravi Kumar", 8, 7, 8, 8, 8, 39],
    ["Lin Wei", 6, 8, 8, 8, 8, 38],
]})

# 14. gradebook -- students by subject
csv("gradebook.csv",
    "Roll,Student,Maths,Science,English,History,Average\n"
    "1,Aarav,88,92,79,71,82.5\n"
    "2,Diya,94,89,91,88,90.5\n"
    "3,Kabir,67,72,80,75,73.5\n"
    "4,Meera,79,85,88,90,85.5\n")

# 15. expense report -- a FORM header plus a TABLE of line items
xl("expense_report.xlsx", {"Claim": [
    ["EXPENSE CLAIM FORM"], [],
    ["Employee:", "Ravi Kumar", "", "Claim No:", "EXP-2291"],
    ["Department:", "Sales", "", "Period:", "Feb 2026"],
    ["Approved by:", "", "", "Status:", "Pending"], [],
    ["Date", "Description", "Category", "Amount"],
    ["2026-02-03", "Client lunch", "Meals", 2400],
    ["2026-02-07", "Taxi to airport", "Travel", 850],
    ["2026-02-11", "Hotel — 2 nights", "Lodging", 11200],
    ["2026-02-18", "Conference fee", "Training", 7500],
    ["", "", "TOTAL", 21950],
]})

# 16. multi-sheet workbook mixing genres
xl("mixed_workbook.xlsx", {
    "Summary": [["Quarterly Summary"], [], ["Metric", "Value"],
                ["Revenue", 1250000], ["Orders", 3420], ["Avg order", 365.5]],
    "Orders": [["order_id", "customer", "amount", "order_date"],
               ["O-1", "Acme", 1200, "2026-01-05"], ["O-2", "Borealis", 890, "2026-01-11"],
               ["O-3", "Cirrus", 2400, "2026-02-02"], ["O-4", "Delta", 640, "2026-02-19"]],
    "Customers": [["customer", "region", "tier"],
                  ["Acme", "North", "Gold"], ["Borealis", "South", "Silver"],
                  ["Cirrus", "North", "Gold"], ["Delta", "East", "Bronze"]],
})
print(f"generated {len(list(D.glob('*')))} files")


# --------------------------------------------------------------------------- #
# Round two: shapes the first pass did not cover
# --------------------------------------------------------------------------- #

# 17. bank statement -- running balance, debit/credit columns, date-first
csv("bank_statement.csv",
    "Txn Date,Value Date,Description,Debit,Credit,Balance\n"
    "01/04/2026,01/04/2026,Opening balance,,,125000.00\n"
    "03/04/2026,03/04/2026,NEFT SALARY ACME LTD,,58000.00,183000.00\n"
    "05/04/2026,05/04/2026,UPI-SWIGGY-4412,640.00,,182360.00\n"
    "07/04/2026,08/04/2026,ATM WDL CHENNAI,5000.00,,177360.00\n"
    "11/04/2026,11/04/2026,RENT TRANSFER,25000.00,,152360.00\n")

# 18. invoice -- header block, line items, tax rows, grand total
xl("invoice.xlsx", {"Invoice": [
    ["TAX INVOICE"], [],
    ["Invoice No:", "INV-2026-0441", "", "Date:", "2026-04-12"],
    ["Bill To:", "Acme Industries", "", "GSTIN:", "33AABCA1234F1Z5"],
    [],
    ["#", "Item", "HSN", "Qty", "Rate", "Amount"],
    [1, "Steel bracket 40mm", "7308", 120, 45.0, 5400.0],
    [2, "Anchor bolt M12", "7318", 500, 12.5, 6250.0],
    [3, "Labour — installation", "9954", 1, 8000.0, 8000.0],
    [], ["", "", "", "", "Subtotal", 19650.0],
    ["", "", "", "", "CGST 9%", 1768.5],
    ["", "", "", "", "SGST 9%", 1768.5],
    ["", "", "", "", "TOTAL", 23187.0],
]})

# 19. attendance register -- days as columns, P/A/L codes
xl("attendance_register.xlsx", {"Apr 2026": [
    ["ATTENDANCE REGISTER — April 2026"], [],
    ["Emp ID", "Name", "1", "2", "3", "4", "5", "Present", "Absent"],
    ["E001", "Asha Menon", "P", "P", "A", "P", "P", 4, 1],
    ["E002", "Ravi Kumar", "P", "L", "P", "P", "P", 4, 0],
    ["E003", "Lin Wei", "A", "A", "P", "P", "P", 3, 2],
]})

# 20. survey responses -- long question headers, Likert values
csv("survey.csv",
    "Respondent,How satisfied are you with the product?,"
    "Would you recommend us to a colleague?,What is your role?,Comments\n"
    "R001,Very satisfied,Yes,Engineer,Works well\n"
    "R002,Neutral,Maybe,Manager,\n"
    "R003,Dissatisfied,No,Analyst,Too slow on large files\n"
    "R004,Satisfied,Yes,Engineer,\n")

# 21. price list -- tiered pricing, merged tier header
xl("price_list.xlsx", {"Prices": [
    ["Wholesale Price List 2026"], [],
    ["SKU", "Product", "Tier 1", "Tier 1", "Tier 2", "Tier 2"],
    ["", "", "Qty", "Price", "Qty", "Price"],
    ["P-100", "Widget A", "1-49", 120.0, "50+", 99.0],
    ["P-101", "Widget B", "1-49", 145.0, "50+", 118.0],
    ["P-102", "Widget C", "1-24", 310.0, "25+", 265.0],
]})

# 22. asset register -- depreciation, dates, mixed types
csv("asset_register.csv",
    "Asset Code,Description,Purchase Date,Cost,Depn Rate,Accum Depn,WDV\n"
    "FA-001,Laptop Dell 5420,2024-06-15,78000,0.40,46800,31200\n"
    "FA-002,Office chair x10,2023-01-10,45000,0.10,13500,31500\n"
    "FA-003,Server rack,2022-11-01,240000,0.15,118800,121200\n"
    "FA-004,Printer HP,2025-03-22,32000,0.40,12800,19200\n")

# 23. crosstab with BOTH row and column totals
csv("crosstab.csv",
    "Product,North,South,East,West,Total\n"
    "Widgets,120,180,90,140,530\n"
    "Gadgets,200,160,210,190,760\n"
    "Gizmos,80,110,70,100,360\n"
    "Total,400,450,370,430,1650\n")

# 24. sheet with formula errors and blanks
csv("with_errors.csv",
    "Item,Qty,Unit Price,Total\n"
    "Bolt,100,2.50,250.00\n"
    "Nut,#N/A,1.20,#VALUE!\n"
    "Washer,200,#DIV/0!,#DIV/0!\n"
    "Screw,50,3.00,150.00\n"
    "Anchor,75,4.20,315.00\n")

# 25. meeting minutes -- mostly prose with an action table at the bottom
xl("minutes.xlsx", {"Minutes": [
    ["PROJECT REVIEW — MINUTES"], [],
    ["Date:", "2026-04-09", "", "Chair:", "R. Kumar"],
    ["Present:", "A. Menon, L. Wei, O. Haddad"], [],
    ["Discussion"], [],
    ["Action", "Owner", "Due", "Status"],
    ["Finalise vendor shortlist", "A. Menon", "2026-04-16", "Open"],
    ["Circulate revised budget", "L. Wei", "2026-04-12", "Done"],
    ["Book site visit", "O. Haddad", "2026-04-20", "Open"],
]})

# 26. multi-currency ledger -- currency symbols mixed in one column
csv("multi_currency.csv",
    "Date,Vendor,Currency,Amount,INR Equivalent\n"
    "2026-01-15,AWS,USD,$1240.50,103562.75\n"
    "2026-01-22,Hetzner,EUR,€89.00,8099.00\n"
    "2026-02-03,Zoho,INR,₹4500.00,4500.00\n"
    "2026-02-14,GitHub,USD,$210.00,17535.00\n")

# 27. wide gantt-ish plan -- week columns with x markers
csv("project_plan.csv",
    "Task,Owner,W1,W2,W3,W4,W5,W6\n"
    "Requirements,Asha,x,x,,,,\n"
    "Design,Lin,,x,x,x,,\n"
    "Build,Ravi,,,x,x,x,\n"
    "Test,Omar,,,,,x,x\n")

# 28. recipe / bill of materials -- nested components
csv("bom.csv",
    "Level,Part No,Description,Qty,UOM\n"
    "1,ASM-100,Pump assembly,1,EA\n"
    "2,PRT-201,Housing,1,EA\n"
    "2,PRT-202,Impeller,1,EA\n"
    "3,RAW-301,Stainless sheet,0.4,KG\n"
    "2,PRT-203,Seal kit,2,SET\n")
