INSTALLATION
- Run database.sql on a DB
- Put all files in a direcorty on the web server.
- Configure the Database and table info in the beginning of index.php / example.php / other files

SAMPLES
- index.php contains minimum code generation
- example-all.php contains few more customization parameters with description
- excel_view.php contains sample how to make navigation similar to excel
- export-pdf.php
- external-link.php
* subgrid.php contains master/detail subgrid example. you can have n-level of subgrid
* custom-events.php contains sample to write you own update/insert/delete code
* export-xlsx.php
* grouping.php


(*) These features are available with full source code, which you can get after appreciating it by little payment (smile). 

See http://www.phpgrid.org/ for more details.

CHANGELOG:

v1.4.6
- Autofilter (dropdown option) (autofilter.php)
- Custom column (with buttons, image, etc) and passing row data as querystring (custom-button.php)
- Custom Layout for Add/Edit Dialogs (custom-layout.php)
- Search on page load (pre-selected filters) (search-onload.php)
- Data Lookup from other table (Combobox) (dropdown.php)
- Alternate Row Coloring (alternate-row.php)
- MSSQL DSN or ODBC sample (db-layer.php)
- Setting Colums for add/edit, not for listing (example-all.php)
- Localization example (localization.php)

Fixes & Minor Additions
- New Events (on_data_display for custom formating, on_after_insert) (custom-events.php)
- Action columns width fix (example-full.php)
- Grouping sample bug fix (short php tag) (grouping.php)
- Several notices and warning fixes
- Excel 2003 support added

v1.4.5 (7:15:34 PM, Sunday, February 12, 2012)
- support for all major databases (mysql,pgsql,mssql,sybase ... and odbc support) (db-layer.php)
- inline row addition (excel_view.php)
- pdf export custom template sample (export-pdf.php)
- export to pdf/excel in same grid
- fix for non integer primary key
- fix for mysql keyword based column name
- Fix for non-integer PK column
- multiselect fix in lib jqgrid 4.3.1


v1.4 (20.12.11)

Features:
- Datepicker integration (example-all.php)
- Group by field (grouping.php)
- Runtime change "group by" field (grouping.php)
- Password mask formatter / edittype
	// To mask password field, apply following attribs
	$col["edittype"] = "password";
	$col["formatter"] = "password";
- Excel export fix for numeric columns

v1.3 (21.07.11)

Features:
- multiple fonts in pdf export
- External link with grid data (external-link.php)
- PDF export feature (export-pdf.php)
- Export filtered/all data option
- Export specific columns (export-pdf.php, $col["export"] = false;)

Fixes:
- persist where filter in export option
- Jquery fix for IE
- utf8 encoding fix
- json fix for php 5.1
- slashes fix


MIGRATION FROM v1.1 TO v1.2 (31.12.10)
- You need to change your grid column name to table-field or field-alias. 
- "table.field" format (i.invdate) is not supported anymore. instead write "invdate"
e.g.	old -- $col["name"] = "i.id"; 
		new -- $col["name"] = "id"; 

FEEDBACK

- Do post bugs/wishlist here ...
http://azgtech.wordpress.com/2010/08/01/jqgrid-php-datagrid-component/

LICENSE

- must read license.txt before use
