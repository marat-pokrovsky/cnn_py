import subprocess
import datetime
import sys
import json

if len(sys.argv) != 4:
    print("Использование: python chg.py <account_name> <token> <repo_name>")
    sys.exit(1)
    
account_name = sys.argv[1]
token = sys.argv[2]
repo_name = sys.argv[3]

# Проверяем и сохраняем несохраненные изменения
status_result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
if status_result.stdout.strip():
    print("Обнаружены несохраненные изменения. Сохраняем их...")
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Auto-save before date modification"])

# Количество дней для сдвига первого коммита
initial_days_shift = 30

# Получаем список всех коммитов (от самого старого к новому)
commits = subprocess.check_output(
    ["git", "log", "--reverse", "--format=%H"],
    text=True
).splitlines()

# Создаем JSON-объект с датами для коммитов
commit_dates = {}
for i, commit in enumerate(commits):
    days_shift = initial_days_shift - i
    new_date = datetime.date.today() - datetime.timedelta(days=days_shift)
    new_date_str = new_date.strftime("%Y-%m-%d %H:%M:%S")
    commit_dates[commit] = new_date_str

# Преобразуем в JSON-строку
commit_dates_json = json.dumps(commit_dates)

# Создаем env-filter скрипт
env_filter_script = f"""
    commit_dates='{commit_dates_json}'
    date=$(echo "$commit_dates" | jq -r --arg commit "$GIT_COMMIT" '.[$commit]')
    if [ -n "$date" ]; then
        export GIT_AUTHOR_DATE="$date"
        export GIT_COMMITTER_DATE="$date"
    fi
"""

# Применяем изменения ко всем коммитам
try:
    subprocess.run(["git", "filter-branch", "-f", "--env-filter", env_filter_script], check=True)
except subprocess.CalledProcessError as e:
    print(f"Ошибка при выполнении filter-branch: {e}")
    sys.exit(1)

# Переименовываем текущую ветку в master
current_branch = subprocess.check_output(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    text=True
).strip()

if current_branch != "main":
    subprocess.run(["git", "branch", "-m", "main"])

# Настройка удаленного репозитория
subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{account_name}/{repo_name}.git"])
subprocess.run(["curl", "-u", f"{account_name}:{token}", "https://api.github.com/user/repos", "-d", f'{{"name":"{repo_name}"}}'])
subprocess.run(["git", "config", f'url.https://{account_name}:{token}@github.com/.insteadOf', 'https://github.com/'])

# Принудительная отправка изменений
subprocess.run(["git", "push", "--force", "--all"])