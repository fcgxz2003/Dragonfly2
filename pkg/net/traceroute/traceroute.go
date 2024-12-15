/*
 *     Copyright 2024 The Dragonfly Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package traceroute

import (
	"log"
	"net"
	"os"
	"time"

	"golang.org/x/net/icmp"
	"golang.org/x/net/ipv4"
)

func SendTracertMsg(dst net.IPAddr, ttl int) (int64, icmp.Type, net.Addr, error) {
	c, err := net.ListenPacket("ip4:1", "0.0.0.0")
	if err != nil {
		return 0, nil, nil, err
	}
	defer c.Close()

	// set ipv4 head
	p := ipv4.NewPacketConn(c)
	if err := p.SetControlMessage(ipv4.FlagTTL|ipv4.FlagSrc|ipv4.FlagDst|ipv4.FlagInterface, true); err != nil {
		return 0, nil, nil, err
	}

	// ICMP message
	wm := icmp.Message{
		Type: ipv4.ICMPTypeEcho,
		Code: 0,
		Body: &icmp.Echo{
			ID:   os.Getpid() & 0xffff,
			Data: []byte("HELLO-R-U-THERE"),
		},
	}

	// IP data
	rb := make([]byte, 1500)

	wm.Body.(*icmp.Echo).Seq = ttl
	wb, err := wm.Marshal(nil)
	if err != nil {
		return 0, nil, nil, err
	}

	// TTL
	if err := p.SetTTL(ttl); err != nil {
		return 0, nil, nil, err
	}

	begin := time.Now()
	if _, err := p.WriteTo(wb, nil, &dst); err != nil {
		return 0, nil, nil, err
	}

	if err := p.SetReadDeadline(time.Now().Add(3 * time.Second)); err != nil {
		return 0, nil, nil, err
	}

	n, _, peer, err := p.ReadFrom(rb)
	if err != nil {

		if err, ok := err.(net.Error); ok && err.Timeout() {
			return 0, ipv4.ICMPTypeDestinationUnreachable, peer, nil
		}
		log.Fatal(err)
	}

	rm, err := icmp.ParseMessage(1, rb[:n])
	if err != nil {
		return 0, nil, nil, err
	}

	rtt := time.Since(begin).Milliseconds()
	return rtt, rm.Type, peer, nil
}

func Tracert4(host string, dst net.IPAddr, maxhoop int) ([]int64, []net.Addr, error) {
	// names, _ := net.LookupAddr(dst.IP.String())
	// if names == nil {
	// 	names = append(names, host)
	// }
	// fmt.Printf("\n通过最多 %v 个跃点跟踪\n到 %v [%s] 的路由:\n\n", maxhoop, names[0], dst.IP)

	var (
		rtts  []int64
		peers []net.Addr
	)

ICMP:
	for i := 1; i <= maxhoop; i++ {
		rtt, icmptype, peer, err := SendTracertMsg(dst, i)
		if err != nil {
			return rtts, peers, err
		}

		switch icmptype {
		case ipv4.ICMPTypeTimeExceeded:
			names, _ := net.LookupAddr(peer.String())
			if names != nil {
				rtts = append(rtts, rtt)
				peers = append(peers, peer)
			}

			continue ICMP
		case ipv4.ICMPTypeEchoReply:
			names, _ := net.LookupAddr(peer.String())
			if names != nil {
				rtts = append(rtts, rtt)
				peers = append(peers, peer)
			}

			break ICMP
		}
	}

	return rtts, peers, nil
}
