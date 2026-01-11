from fastmcp import FastMCP

from fastmcp import FastMCP
import psycopg2
import asyncpg
from datetime import datetime
import json
import re
import langchain

import langchain_huggingface 
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint, HuggingFaceEmbeddings
import os 

from dotenv import load_dotenv
load_dotenv()
def create_model():
    
   hf_token=os.getenv("HF_TOKEN")
   if not hf_token:
        raise ValueError("the hf token is not available")   
   repo_id="Qwen/Qwen2.5-7B-Instruct"     
   llm=HuggingFaceEndpoint(
       repo_id=repo_id,
       huggingfacehub_api_token=hf_token,
       task="conversational"
   ) 
   model=ChatHuggingFace(llm=llm)

   return model 



mcp = FastMCP(name="ProductivityMCP")

# --------------------------
# Database Initialization
# --------------------------
def initialise_db():
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )
    cur = conn.cursor()

    cur.execute("""
    CREATE SCHEMA IF NOT EXISTS productivity;
    """)

 
    cur.execute("""
     CREATE TABLE if not exists productivity.productivity_assessments (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(50) NOT NULL,
    recommended_action VARCHAR(50) NOT NULL,
    confidence NUMERIC(3,2) CHECK(confidence BETWEEN 0 AND 1)
);

    
    """)

    conn.commit()
    cur.close()
    conn.close()


async def list_tasks2(filters: dict = {}):
    
    conditions = []
    values = []

    if "priority" in filters:
        conditions.append(f"priority=${len(values)+1}")
        values.append(filters["priority"])

    if "status" in filters:
        conditions.append(f"status=${len(values)+1}")
        values.append(filters["status"])

    if "due_date" in filters:
        due_date = datetime.strptime(filters["due_date"], "%Y-%m-%d").date()
        conditions.append(f"due_date=${len(values)+1}")
        values.append(due_date)

    if "due_date_start" in filters:
        start = datetime.strptime(filters["due_date_start"], "%Y-%m-%d").date()
        conditions.append(f"due_date>=${len(values)+1}")
        values.append(start)

    if "due_date_end" in filters:
        end = datetime.strptime(filters["due_date_end"], "%Y-%m-%d").date()
        conditions.append(f"due_date<=${len(values)+1}")
        values.append(end)

    if "keyword" in filters:
        conditions.append(f"task_name LIKE ${len(values)+1}")
        values.append(f"%{filters['keyword']}%")

    sql = "SELECT * FROM tasks WHERE " + " AND ".join(conditions)

    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="projectmanagement"
    )

    rows = await conn.fetch(sql, *values)
    await conn.close()

    return {
        "message": "Tasks fetched successfully.",
        "rows": [dict(row) for row in rows]
    }

@mcp.tool

async def summary(filters: dict = {}):
    """
    Return a summary of tasks, including total, pending, completed, overdue.
    Optional filters: priority, status, due_date
    """
   
    result = await list_tasks2(filters)
    rows = result.get('rows', [])

    total = len(rows)
    pending = 0
    completed = 0
    overdue = 0
    high_priority=0
    low_priority=0
    medium_priority=0
    today = datetime.today().date()
    x=datetime.strptime(filters['due_date'],"%Y-%m-%d").date()
    filters['due_date']=x
    for row in rows:
        priority=row['priority']
        due_date = row['due_date']  # access dict key
        status = row['status']
        if priority=='high':
            high_priority+=1
        elif priority=='low':
            low_priority+=1
        else:
            medium_priority+=1        
        

        if status == 'pending':
            pending += 1
            if due_date:
                if isinstance(due_date, str):
                    # If somehow a string, convert to date
                    try:
                        due_date = datetime.strptime(due_date, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                if due_date < today:
                    overdue += 1
        elif status == 'completed':
            completed += 1

    data= {
        "total": total,
        "pending": pending,
        "completed": completed,
        "overdue": overdue,
        "high_priority_tasks_percentage":high_priority/total,
        "low_priority_tasks_percentage":low_priority/total,
        "medium_priority_tasks_percentage":medium_priority/total
    }
    model = create_model()

    prompt = f"""
You are a productivity analyzing machine.

Return ONLY valid JSON with keys:
- state
- action
- confidence

Allowed states:
-on_track → most tasks completed, nothing overdue
-behind_schedule → some overdue/pending high-priority tasks
-overloaded → too many high-priority tasks pending 

Allowed actions:
-focus → work on urgent/high-priority tasks
-reschedule → postpone lower-priority tasks
-delegate → assign tasks to others if possible
-maintain → keep current pace


Confidence:
- number between 0 and 1 signifying the ai confidence in action

Input metrics:
{data}
"""

    response = model.invoke(prompt)
    text = response.content

    clean = re.sub(r"```json|```", "", text).strip()
    output = json.loads(clean)
    conn = await asyncpg.connect(
        host="localhost",
        port=5432,
        user="postgres",
        password="kunal",
        database="user_info"
    )

    await conn.execute(
        """
        INSERT INTO productivity.productivity_assessments
        (state, recommended_action, confidence, date)
        VALUES ($1,$2,$3,$4)
        """,
        output["state"],
        output["action"],
        output["confidence"],
        filters.get('due_date',datetime.today().date())
        
    )

    await conn.close()

    return {
        "date":filters.get('due_date',datetime.today().date()),
        
        "signal": output
    }

initialise_db()
if __name__=="__main__":
    mcp.run(transport='http',port=8000,host='0.0.0.0')



