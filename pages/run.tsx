/* eslint-disable @typescript-eslint/naming-convention */
import React from 'react'
import { GetServerSideProps } from 'next'
import fetch from 'isomorphic-fetch'
import { format, subDays } from 'date-fns'
import qs from 'query-string'
import { Line } from 'react-chartjs-2'

import Layout from '#components/Layout'
import Sidebar from '#components/Sidebar/Sidebar'
import Page from '#components/Page'

const HOST_URL = process.env.HOST_URL

type DATE_TYPE = Array<{ x: string; y: number }>

export default function Run({ data }: { data: DATE_TYPE }) {
  return (
    <Layout title="Daily Running (Beta)" url={`${HOST_URL}/run`}>
      <Sidebar />
      <Page title="Daily Running (Beta)">
        <div style={{ height: '70vh', overflowX: 'scroll' }}>
          <div style={{ width: '1500px' }}>
            <Line
              width={100}
              height={30}
              data={{
                labels: data.map((d) => d.x),
                datasets: [
                  {
                    label: 'daily',
                    data: data.map((d) => d.y),
                  },
                ],
              }}
              options={{
                scales: {
                  yAxes: [
                    {
                      ticks: {
                        beginAtZero: true,
                      },
                    },
                  ],
                  xAxes: [
                    {
                      type: 'time',
                      time: {
                        parser: 'YYYY-MM-DD',
                        unit: 'day',
                        displayFormats: {
                          minute: 'YYYY-MM-DD',
                        },
                      },
                    },
                  ],
                },
              }}
            />
          </div>
        </div>
      </Page>
    </Layout>
  )
}

export const getServerSideProps: GetServerSideProps = async (context) => {
  const defaultStartDate = format(subDays(new Date(), 30), 'yyyy-MM-dd')
  const defaultEndDate = format(new Date(), 'yyyy-MM-dd')

  const {
    startDate = defaultStartDate,
    endDate = defaultEndDate,
  } = context.query

  const query = qs.stringify({ start_date: startDate, end_date: endDate })

  const response = await fetch(`${HOST_URL}/api/run?${query}`)
  const data: Array<{
    run: number
    date: string
    id: string
  }> = await response.json()

  const result: DATE_TYPE = data.map(({ run, date }) => ({
    y: Math.round(run * 100) / 100,
    x: format(new Date(date), 'yyyy-MM-dd'),
  }))

  return {
    props: {
      data: result,
    },
  }
}
