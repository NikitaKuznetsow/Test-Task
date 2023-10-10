Final SQL Statement 
'''
WITH cnt_flights AS (
    SELECT
        Id_Psg,
        COUNT(Trip_No) AS cnt
    FROM
        main.Pass_In_Trip
    GROUP BY
        Id_Psg
),
names AS (
    SELECT
        Id_Psg,
        Name,
        COALESCE(LAG(Id_Psg) OVER w2, LAST_VALUE(Id_Psg) OVER w1) AS prev,
        COALESCE(LEAD(Id_Psg) OVER w2, FIRST_VALUE(Id_Psg) OVER w1) AS next
    FROM
        main.Passenger AS P
    WINDOW
        w1 AS (
            ORDER BY
                Id_Psg ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ),
        w2 AS (
            ORDER BY
                Id_Psg
        )
)

SELECT
    names.Name,
    (
        SELECT
            Name
        FROM
            main.Passenger AS P
        WHERE
            P.Id_Psg = names.prev
    ) AS Previous,
    (
        SELECT
            Name
        FROM
            main.Passenger AS P
        WHERE
            P.Id_Psg = names.next
    ) AS Next
FROM
    cnt_flights
JOIN
    names USING(Id_Psg)
WHERE
    cnt = (
        SELECT
            MAX(cnt)
        FROM
            cnt_flights
    );


'''