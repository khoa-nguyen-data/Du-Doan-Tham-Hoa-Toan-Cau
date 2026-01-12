# restore_notebooks.ps1
# Script khôi phục các file notebook bị hỏng từ commit cũ
# Thay thế file hiện tại (bị hỏng) bằng phiên bản từ commit cũ (còn hợp lệ)
# Không thay đổi HEAD hay branch hiện tại

param(
    [string]$ProjectPath = (Get-Location).Path
)

# Chuyển đến thư mục project
Set-Location $ProjectPath

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SCRIPT KHÔI PHỤC FILE NOTEBOOK BỊ HỎNG" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Hàm kiểm tra file JSON có hợp lệ không
function Test-ValidJson {
    param([string]$FilePath)
    try {
        $content = Get-Content $FilePath -Raw -ErrorAction Stop
        $null = $content | ConvertFrom-Json -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Hàm tìm commit cũ có file hợp lệ
function Find-ValidCommit {
    param([string]$FilePath)
    
    # Lấy danh sách commit của file (tối đa 20 commit gần nhất)
    $commits = git log --oneline --all -n 20 -- $FilePath 2>$null
    
    if (-not $commits) {
        return $null
    }
    
    foreach ($line in $commits) {
        $commitHash = ($line -split " ")[0]
        
        # Thử lấy nội dung file từ commit và kiểm tra JSON
        try {
            $content = git show "${commitHash}:${FilePath}" 2>$null
            if ($content) {
                $null = $content | ConvertFrom-Json -ErrorAction Stop
                return $commitHash
            }
        } catch {
            # Commit này không có file hợp lệ, tiếp tục tìm
            continue
        }
    }
    return $null
}

# Hàm khôi phục file từ commit
function Restore-NotebookFile {
    param(
        [string]$FilePath,
        [string]$CommitHash
    )
    
    Write-Host "  Đang khôi phục từ commit: $CommitHash" -ForegroundColor Yellow
    git checkout $CommitHash -- $FilePath 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Khôi phục thành công!" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ✗ Khôi phục thất bại!" -ForegroundColor Red
        return $false
    }
}

# Tìm tất cả file .ipynb trong project
Write-Host "Đang quét các file notebook (.ipynb)..." -ForegroundColor White
$notebookFiles = Get-ChildItem -Path $ProjectPath -Filter "*.ipynb" -Recurse -File

$totalFiles = $notebookFiles.Count
$corruptedFiles = 0
$restoredFiles = 0
$failedFiles = 0

Write-Host "Tìm thấy $totalFiles file notebook" -ForegroundColor White
Write-Host ""

foreach ($file in $notebookFiles) {
    $relativePath = $file.FullName.Replace("$ProjectPath\", "").Replace("\", "/")
    
    Write-Host "Kiểm tra: $relativePath" -ForegroundColor White
    
    if (Test-ValidJson -FilePath $file.FullName) {
        Write-Host "  ✓ File hợp lệ" -ForegroundColor Green
    } else {
        $corruptedFiles++
        Write-Host "  ✗ File bị hỏng JSON!" -ForegroundColor Red
        
        # Tìm commit có file hợp lệ
        Write-Host "  Đang tìm commit có file hợp lệ..." -ForegroundColor Yellow
        $validCommit = Find-ValidCommit -FilePath $relativePath
        
        if ($validCommit) {
            $restored = Restore-NotebookFile -FilePath $relativePath -CommitHash $validCommit
            if ($restored) {
                $restoredFiles++
            } else {
                $failedFiles++
            }
        } else {
            Write-Host "  ✗ Không tìm thấy commit hợp lệ!" -ForegroundColor Red
            $failedFiles++
        }
    }
    Write-Host ""
}

# Báo cáo kết quả
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "              KẾT QUẢ" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "Tổng số file:        $totalFiles" -ForegroundColor White
Write-Host "File bị hỏng:        $corruptedFiles" -ForegroundColor $(if ($corruptedFiles -gt 0) { "Yellow" } else { "Green" })
Write-Host "Đã khôi phục:        $restoredFiles" -ForegroundColor $(if ($restoredFiles -gt 0) { "Green" } else { "White" })
Write-Host "Thất bại:            $failedFiles" -ForegroundColor $(if ($failedFiles -gt 0) { "Red" } else { "Green" })
Write-Host "============================================" -ForegroundColor Cyan

if ($restoredFiles -gt 0) {
    Write-Host ""
    Write-Host "LƯU Ý: Các file đã được khôi phục trong working directory." -ForegroundColor Yellow
    Write-Host "Bạn có thể commit các thay đổi này nếu muốn lưu lại." -ForegroundColor Yellow
}
