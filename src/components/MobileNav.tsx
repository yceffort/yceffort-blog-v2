'use client'

import Link from 'next/link'
import {memo, useState} from 'react'

import {SiteConfig} from '#src/config'

const MobileNav = memo(() => {
    const [navShow, setNavShow] = useState(false)

    const onToggleNav = () => {
        setNavShow((status) => {
            if (status) {
                document.body.style.overflow = 'auto'
            } else {
                // Prevent scrolling
                document.body.style.overflow = 'hidden'
            }
            return !status
        })
    }

    return (
        <div className="sm:hidden">
            <button type="button" className="ml-1 mr-1 h-8 w-8 rounded" aria-label="Toggle Menu" onClick={onToggleNav}>
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    className="text-gray-900 dark:text-gray-100"
                >
                    {navShow ? (
                        <path
                            fillRule="evenodd"
                            d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                            clipRule="evenodd"
                        />
                    ) : (
                        <path
                            fillRule="evenodd"
                            d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
                            clipRule="evenodd"
                        />
                    )}
                </svg>
            </button>
            <div
                className={`fixed top-24 right-0 z-10 h-full w-full transform bg-gray-200 opacity-95 duration-300 ease-in-out dark:bg-gray-800 ${
                    navShow ? 'translate-x-0' : 'translate-x-full'
                }`}
            >
                <button
                    type="button"
                    aria-label="toggle modal"
                    className="fixed h-full w-full cursor-auto focus:outline-none"
                    onClick={onToggleNav}
                />
                <nav className="fixed mt-8 h-full">
                    {SiteConfig.menu.map((link) => (
                        <div key={link.path} className="px-12 py-4">
                            <Link
                                href={link.path}
                                className="text-2xl font-bold tracking-widest text-gray-900 dark:text-gray-100"
                                onClick={onToggleNav}
                            >
                                {link.label}
                            </Link>
                        </div>
                    ))}
                </nav>
            </div>
        </div>
    )
})

export default MobileNav
