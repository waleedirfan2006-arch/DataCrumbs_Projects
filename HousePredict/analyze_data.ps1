$csvPath = 'c:/Users/Teslas Laptops/OneDrive/Desktop/DataCrumbs/DataCrumbs_Projects/HousePredict/data/data.csv'
$csv = Import-Csv -Path $csvPath -Delimiter ','

# Number of rows
$rows = $csv.Count

# Column names
$columns = $csv[0].psobject.Properties.Name

# Determine numeric vs categorical based on first non-empty value
$numericCols = @()
$categoricalCols = @()
foreach ($col in $columns) {
    $firstNonEmpty = $csv | Where-Object { $_.$col -ne $null -and $_.$col -ne '' } | Select-Object -First 1
    if ($firstNonEmpty) {
        $value = $firstNonEmpty.$col
        if ($value -as [double]) {
            $numericCols += $col
        } else {
            $categoricalCols += $col
        }
    } else {
        # If all values empty, treat as categorical
        $categoricalCols += $col
    }
}

# Missing values per column
$missingPerCol = @{}
foreach ($col in $columns) {
    $missingCount = ($csv | Where-Object { -not $_.$col -or $_.$col -eq '' }).Count
    $missingPerCol[$col] = $missingCount
}

# Duplicate rows count
$rowStrings = $csv | ForEach-Object { $_ | ConvertTo-Json -Compress }
$duplicateCount = ($rowStrings | Group-Object | Where-Object { $_.Count -gt 1 }).Count

# Print summary
Write-Host "Rows: $rows"
Write-Host "Columns: $($columns.Count)"
Write-Host "Column names: $($columns -join ', ')"
Write-Host "Numeric columns: $($numericCols -join ', ')"
Write-Host "Categorical columns: $($categoricalCols -join ', ')"
Write-Host "Missing values per column:"
foreach ($col in $columns) {
    Write-Host "  $col : $($missingPerCol[$col])"
}
Write-Host "Duplicate rows: $duplicateCount"
